import asyncio
import json
import logging
import os
import pickle
import socket
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type
import grpc
import starlette
import starlette.routing
import uvicorn
from packaging import version
from starlette.datastructures import MutableHeaders
from starlette.middleware import Middleware
from starlette.types import Receive
import ray
from ray import serve
from ray._private.utils import get_or_create_event_loop
from ray.actor import ActorHandle
from ray.serve._private.common import EndpointInfo, EndpointTag, NodeId, RequestProtocol
from ray.serve._private.constants import (
from ray.serve._private.grpc_util import DummyServicer, create_serve_grpc_server
from ray.serve._private.http_util import (
from ray.serve._private.logging_utils import (
from ray.serve._private.long_poll import LongPollClient, LongPollNamespace
from ray.serve._private.proxy_request_response import (
from ray.serve._private.proxy_response_generator import ProxyResponseGenerator
from ray.serve._private.proxy_router import (
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import call_function_from_import_path
from ray.serve.config import gRPCOptions
from ray.serve.generated.serve_pb2 import HealthzResponse, ListApplicationsResponse
from ray.serve.generated.serve_pb2_grpc import add_RayServeAPIServiceServicer_to_server
from ray.serve.handle import DeploymentHandle
from ray.serve.schema import LoggingConfig
from ray.util import metrics
class HTTPProxy(GenericProxy):
    """This class is meant to be instantiated and run by an ASGI HTTP server.

    >>> import uvicorn
    >>> controller_name = ... # doctest: +SKIP
    >>> uvicorn.run(HTTPProxy(controller_name)) # doctest: +SKIP
    """

    @property
    def protocol(self) -> RequestProtocol:
        return RequestProtocol.HTTP

    async def not_found(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        status_code = 404
        for message in convert_object_to_asgi_messages(f"Path '{proxy_request.path}' not found. Please ping http://.../-/routes for route table.", status_code=status_code):
            yield message
        yield ResponseStatus(code=status_code, is_error=True)

    async def draining_response(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        status_code = 503
        for message in convert_object_to_asgi_messages(DRAINED_MESSAGE, status_code=status_code):
            yield message
        yield ResponseStatus(code=status_code, is_error=True)

    async def timeout_response(self, proxy_request: ProxyRequest, request_id: str) -> ResponseGenerator:
        status_code = 408
        for message in convert_object_to_asgi_messages(f'Request {request_id} timed out after {self.request_timeout_s}s.', status_code=status_code):
            yield message
        yield ResponseStatus(code=status_code, is_error=True)

    async def routes_response(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        status_code = 200
        routes_dict = dict()
        for route, endpoint in self.route_info.items():
            if endpoint.app:
                routes_dict[route] = endpoint.app
            else:
                routes_dict[route] = endpoint.name
        for message in convert_object_to_asgi_messages(routes_dict, status_code=status_code):
            yield message
        yield ResponseStatus(code=status_code)

    async def health_response(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        status_code = 200
        for message in convert_object_to_asgi_messages(HEALTH_CHECK_SUCCESS_MESSAGE, status_code=status_code):
            yield message
        yield ResponseStatus(code=status_code)

    async def receive_asgi_messages(self, request_id: str) -> ResponseGenerator:
        queue = self.asgi_receive_queues.get(request_id, None)
        if queue is None:
            raise KeyError(f'Request ID {request_id} not found.')
        await queue.wait_for_message()
        return queue.get_messages_nowait()

    async def __call__(self, scope, receive, send):
        """Implements the ASGI protocol.

        See details at:
            https://asgi.readthedocs.io/en/latest/specs/index.html.
        """
        proxy_request = ASGIProxyRequest(scope=scope, receive=receive, send=send)
        async for message in self.proxy_request(proxy_request):
            if not isinstance(message, ResponseStatus):
                await send(message)

    async def proxy_asgi_receive(self, receive: Receive, queue: ASGIMessageQueue) -> Optional[int]:
        """Proxies the `receive` interface, placing its messages into the queue.

        Once a disconnect message is received, the call exits and `receive` is no longer
        called.

        For HTTP messages, `None` is always returned.
        For websocket messages, the disconnect code is returned if a disconnect code is
        received.
        """
        try:
            while True:
                msg = await receive()
                await queue(msg)
                if msg['type'] == 'http.disconnect':
                    return None
                if msg['type'] == 'websocket.disconnect':
                    return msg['code']
        finally:
            queue.close()

    def setup_request_context_and_handle(self, app_name: str, handle: DeploymentHandle, route_path: str, proxy_request: ProxyRequest) -> Tuple[DeploymentHandle, str]:
        """Setup request context and handle for the request.

        Unpack HTTP request headers and extract info to set up request context and
        handle.
        """
        request_context_info = {'route': route_path, 'app_name': app_name}
        for key, value in proxy_request.headers:
            if key.decode() == SERVE_MULTIPLEXED_MODEL_ID:
                multiplexed_model_id = value.decode()
                handle = handle.options(multiplexed_model_id=multiplexed_model_id)
                request_context_info['multiplexed_model_id'] = multiplexed_model_id
            if key.decode() == 'x-request-id':
                request_context_info['request_id'] = value.decode()
        ray.serve.context._serve_request_context.set(ray.serve.context._RequestContext(**request_context_info))
        return (handle, request_context_info['request_id'])

    async def _format_handle_arg_for_java(self, proxy_request: ProxyRequest) -> bytes:
        """Convert an HTTP request to the Java-accepted format (single byte string)."""
        query_string = proxy_request.scope.get('query_string')
        http_body_bytes = await receive_http_body(proxy_request.scope, proxy_request.receive, proxy_request.send)
        if query_string:
            arg = query_string.decode().split('=', 1)[1]
        else:
            arg = http_body_bytes.decode()
        return arg

    async def send_request_to_replica(self, request_id: str, handle: DeploymentHandle, proxy_request: ProxyRequest, app_is_cross_language: bool=False) -> ResponseGenerator:
        """Send the request to the replica and yield its response messages.

        The yielded values will be ASGI messages until the final one, which will be
        the status code.
        """
        if app_is_cross_language:
            handle_arg = await self._format_handle_arg_for_java(proxy_request)
            result_callback = convert_object_to_asgi_messages
        else:
            handle_arg = proxy_request.request_object(proxy_handle=self.self_actor_handle)
            result_callback = pickle.loads
        receive_queue = ASGIMessageQueue()
        self.asgi_receive_queues[request_id] = receive_queue
        proxy_asgi_receive_task = get_or_create_event_loop().create_task(self.proxy_asgi_receive(proxy_request.receive, receive_queue))
        response_generator = ProxyResponseGenerator(handle.remote(handle_arg), timeout_s=self.request_timeout_s, disconnected_task=proxy_asgi_receive_task, result_callback=result_callback)
        status: Optional[ResponseStatus] = None
        response_started = False
        expecting_trailers = False
        try:
            async for asgi_message_batch in response_generator:
                for asgi_message in asgi_message_batch:
                    if asgi_message['type'] == 'http.response.start':
                        status_code = str(asgi_message['status'])
                        status = ResponseStatus(code=status_code, is_error=status_code != '200')
                        expecting_trailers = asgi_message.get('trailers', False)
                    elif asgi_message['type'] == 'websocket.accept':
                        response_generator.stop_checking_for_disconnect()
                    elif asgi_message['type'] == 'http.response.body' and (not asgi_message.get('more_body', False)) and (not expecting_trailers):
                        response_generator.stop_checking_for_disconnect()
                    elif asgi_message['type'] == 'http.response.trailers':
                        if not asgi_message.get('more_trailers', False):
                            response_generator.stop_checking_for_disconnect()
                    elif asgi_message['type'] == 'websocket.disconnect':
                        status = ResponseStatus(code=str(asgi_message['code']), is_error=False)
                        response_generator.stop_checking_for_disconnect()
                    yield asgi_message
                    response_started = True
        except TimeoutError:
            status = ResponseStatus(code=TIMEOUT_ERROR_CODE, is_error=True)
            logger.warning(f'Request {request_id} timed out after {self.request_timeout_s}s.')
            if not response_started:
                async for message in self.timeout_response(proxy_request, request_id):
                    yield message
        except asyncio.CancelledError:
            status = ResponseStatus(code=DISCONNECT_ERROR_CODE, is_error=True)
            logger.info(f'Client for request {request_id} disconnected, cancelling request.')
        except Exception as e:
            logger.exception(e)
            status = ResponseStatus(code='500', is_error=True)
        finally:
            receive_client_disconnect_msg = False
            if not proxy_asgi_receive_task.done():
                proxy_asgi_receive_task.cancel()
            else:
                receive_client_disconnect_msg = True
            if status is None and proxy_request.request_type == 'websocket':
                if receive_client_disconnect_msg:
                    status = ResponseStatus(code=str(proxy_asgi_receive_task.result()), is_error=True)
                else:
                    status = ResponseStatus(code='1000', is_error=True)
            del self.asgi_receive_queues[request_id]
        assert status is not None
        yield status
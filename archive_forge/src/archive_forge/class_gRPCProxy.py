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
class gRPCProxy(GenericProxy):
    """This class is meant to be instantiated and run by an gRPC server.

    This is the servicer class for the gRPC server. It implements `unary_unary`
    as the entry point for unary gRPC request and `unary_stream` as the entry
    point for streaming gRPC request.
    """

    @property
    def protocol(self) -> RequestProtocol:
        return RequestProtocol.GRPC

    async def not_found(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        if not proxy_request.app_name:
            application_message = 'Application metadata not set.'
        else:
            application_message = f"Application '{proxy_request.app_name}' not found."
        not_found_message = f'{application_message} Please ping /ray.serve.RayServeAPIService/ListApplications for available applications.'
        yield ResponseStatus(code=grpc.StatusCode.NOT_FOUND, message=not_found_message, is_error=True)

    async def draining_response(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        if proxy_request.is_route_request:
            application_names = [endpoint.app for endpoint in self.route_info.values()]
            response_proto = ListApplicationsResponse(application_names=application_names)
        else:
            response_proto = HealthzResponse(message=DRAINED_MESSAGE)
        yield response_proto.SerializeToString()
        yield ResponseStatus(code=grpc.StatusCode.UNAVAILABLE, message=DRAINED_MESSAGE, is_error=True)

    async def routes_response(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        application_names = [endpoint.app for endpoint in self.route_info.values()]
        yield ListApplicationsResponse(application_names=application_names).SerializeToString()
        yield ResponseStatus(code=grpc.StatusCode.OK, message=HEALTH_CHECK_SUCCESS_MESSAGE)

    async def health_response(self, proxy_request: ProxyRequest) -> ResponseGenerator:
        yield HealthzResponse(message=HEALTH_CHECK_SUCCESS_MESSAGE).SerializeToString()
        yield ResponseStatus(code=grpc.StatusCode.OK, message=HEALTH_CHECK_SUCCESS_MESSAGE)

    def service_handler_factory(self, service_method: str, stream: bool) -> Callable:

        def set_grpc_code_and_details(context: grpc._cython.cygrpc._ServicerContext, status: ResponseStatus):
            if not context.code():
                context.set_code(status.code)
            if not context.details():
                context.set_details(status.message)

        async def unary_unary(request_proto: Any, context: grpc._cython.cygrpc._ServicerContext) -> bytes:
            """Entry point of the gRPC proxy unary request.

            This method is called by the gRPC server when a unary request is received.
            It wraps the request in a ProxyRequest object and calls proxy_request.
            The return value is serialized user defined protobuf bytes.
            """
            proxy_request = gRPCProxyRequest(request_proto=request_proto, context=context, service_method=service_method, stream=False)
            status = None
            response = None
            async for message in self.proxy_request(proxy_request=proxy_request):
                if isinstance(message, ResponseStatus):
                    status = message
                else:
                    response = message
            set_grpc_code_and_details(context, status)
            return response

        async def unary_stream(request_proto: Any, context: grpc._cython.cygrpc._ServicerContext) -> Generator[bytes, None, None]:
            """Entry point of the gRPC proxy streaming request.

            This method is called by the gRPC server when a streaming request is
            received. It wraps the request in a ProxyRequest object and calls
            proxy_request. The return value is a generator of serialized user defined
            protobuf bytes.
            """
            proxy_request = gRPCProxyRequest(request_proto=request_proto, context=context, service_method=service_method, stream=True)
            status = None
            async for message in self.proxy_request(proxy_request=proxy_request):
                if isinstance(message, ResponseStatus):
                    status = message
                else:
                    yield message
            set_grpc_code_and_details(context, status)
        return unary_stream if stream else unary_unary

    def setup_request_context_and_handle(self, app_name: str, handle: DeploymentHandle, route_path: str, proxy_request: ProxyRequest) -> Tuple[DeploymentHandle, str]:
        """Setup request context and handle for the request.

        Unpack gRPC request metadata and extract info to set up request context and
        handle.
        """
        multiplexed_model_id = proxy_request.multiplexed_model_id
        request_id = proxy_request.request_id
        if not request_id:
            request_id = generate_request_id()
            proxy_request.request_id = request_id
        handle = handle.options(stream=proxy_request.stream, multiplexed_model_id=multiplexed_model_id, method_name=proxy_request.method_name)
        request_context_info = {'route': route_path, 'request_id': request_id, 'app_name': app_name, 'multiplexed_model_id': multiplexed_model_id, 'grpc_context': proxy_request.ray_serve_grpc_context}
        ray.serve.context._serve_request_context.set(ray.serve.context._RequestContext(**request_context_info))
        proxy_request.send_request_id(request_id=request_id)
        return (handle, request_id)

    async def send_request_to_replica(self, request_id: str, handle: DeploymentHandle, proxy_request: ProxyRequest, app_is_cross_language: bool=False) -> ResponseGenerator:
        handle_arg = proxy_request.request_object(proxy_handle=self.self_actor_handle)
        response_generator = ProxyResponseGenerator(handle.remote(handle_arg), timeout_s=self.request_timeout_s)
        try:
            async for context, result in response_generator:
                context.set_on_grpc_context(proxy_request.context)
                yield result
            yield ResponseStatus(code=grpc.StatusCode.OK)
        except TimeoutError:
            message = f'Request {request_id} timed out after {self.request_timeout_s}s.'
            logger.warning(message)
            yield ResponseStatus(code=grpc.StatusCode.DEADLINE_EXCEEDED, is_error=True, message=message)
        except asyncio.CancelledError:
            message = f'Client for request {request_id} disconnected.'
            logger.info(message)
            yield ResponseStatus(code=grpc.StatusCode.CANCELLED, is_error=True, message=message)
        except Exception as e:
            logger.exception(e)
            yield ResponseStatus(code=grpc.StatusCode.INTERNAL, is_error=True, message=str(e))
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
@ray.remote(num_cpus=0)
class ProxyActor:

    def __init__(self, host: str, port: int, root_path: str, controller_name: str, node_ip_address: str, node_id: NodeId, logging_config: LoggingConfig, request_timeout_s: Optional[float]=None, http_middlewares: Optional[List['starlette.middleware.Middleware']]=None, keep_alive_timeout_s: int=DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_S, grpc_options: Optional[gRPCOptions]=None, long_poll_client: Optional[LongPollClient]=None):
        self.grpc_options = grpc_options or gRPCOptions()
        self.long_poll_client = long_poll_client or LongPollClient(ray.get_actor(controller_name, namespace=SERVE_NAMESPACE), {LongPollNamespace.GLOBAL_LOGGING_CONFIG: self._update_logging_config}, call_in_event_loop=get_or_create_event_loop())
        configure_component_logger(component_name='proxy', component_id=node_ip_address, logging_config=logging_config)
        logger.info(f'Proxy actor {ray.get_runtime_context().get_actor_id()} starting on node {node_id}.')
        logger.debug(f'Congiure Porxy actor {ray.get_runtime_context().get_actor_id()} logger with logging config: {logging_config}')
        configure_component_memory_profiler(component_name='proxy', component_id=node_ip_address)
        self.cpu_profiler, self.cpu_profiler_log = configure_component_cpu_profiler(component_name='proxy', component_id=node_ip_address)
        if http_middlewares is None:
            http_middlewares = [Middleware(RequestIdMiddleware)]
        else:
            http_middlewares.append(Middleware(RequestIdMiddleware))
        if RAY_SERVE_HTTP_PROXY_CALLBACK_IMPORT_PATH:
            logger.info(f'Calling user-provided callback from import path  {RAY_SERVE_HTTP_PROXY_CALLBACK_IMPORT_PATH}.')
            middlewares = validate_http_proxy_callback_return(call_function_from_import_path(RAY_SERVE_HTTP_PROXY_CALLBACK_IMPORT_PATH))
            http_middlewares.extend(middlewares)
        self.host = host
        self.port = port
        self.grpc_port = self.grpc_options.port
        self.root_path = root_path
        self.keep_alive_timeout_s = RAY_SERVE_HTTP_KEEP_ALIVE_TIMEOUT_S or keep_alive_timeout_s
        self._uvicorn_server = None
        self.node_ip_address = node_ip_address
        self.http_setup_complete = asyncio.Event()
        self.grpc_setup_complete = asyncio.Event()
        self.http_proxy = HTTPProxy(controller_name=controller_name, node_id=node_id, node_ip_address=node_ip_address, proxy_router_class=LongestPrefixRouter, request_timeout_s=request_timeout_s or RAY_SERVE_REQUEST_PROCESSING_TIMEOUT_S)
        self.grpc_proxy = gRPCProxy(controller_name=controller_name, node_id=node_id, node_ip_address=node_ip_address, proxy_router_class=EndpointRouter, request_timeout_s=request_timeout_s or RAY_SERVE_REQUEST_PROCESSING_TIMEOUT_S) if self.should_start_grpc_service() else None
        self.wrapped_http_proxy = self.http_proxy
        for middleware in http_middlewares:
            if version.parse(starlette.__version__) < version.parse('0.35.0'):
                self.wrapped_http_proxy = middleware.cls(self.wrapped_http_proxy, **middleware.options)
            else:
                self.wrapped_http_proxy = middleware.cls(self.wrapped_http_proxy, *middleware.args, **middleware.kwargs)
        self.running_task_http = get_or_create_event_loop().create_task(self.run_http_server())
        self.running_task_grpc = get_or_create_event_loop().create_task(self.run_grpc_server())

    def _update_logging_config(self, logging_config: LoggingConfig):
        configure_component_logger(component_name='proxy', component_id=self.node_ip_address, logging_config=logging_config)

    def _get_logging_config(self) -> Tuple:
        """Get the logging configuration (for testing purposes)."""
        log_file_path = None
        for handler in logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                log_file_path = handler.baseFilename
        return log_file_path

    def should_start_grpc_service(self) -> bool:
        """Determine whether gRPC service should be started.

        gRPC service will only be started if a valid port is provided and if the
        servicer functions are passed.
        """
        return self.grpc_port > 0 and len(self.grpc_options.grpc_servicer_functions) > 0

    async def ready(self):
        """Returns when both HTTP and gRPC proxies are ready to serve traffic.
        Or throw exception when either proxy is not able to serve traffic.
        """
        http_setup_complete_wait_task = get_or_create_event_loop().create_task(self.http_setup_complete.wait())
        grpc_setup_complete_wait_task = get_or_create_event_loop().create_task(self.grpc_setup_complete.wait())
        waiting_tasks_http = [http_setup_complete_wait_task, self.running_task_http]
        done_set_http, _ = await asyncio.wait(waiting_tasks_http, return_when=asyncio.FIRST_COMPLETED)
        waiting_tasks_grpc = [grpc_setup_complete_wait_task, self.running_task_grpc]
        done_set_grpc, _ = await asyncio.wait(waiting_tasks_grpc, return_when=asyncio.FIRST_COMPLETED)
        if self.http_setup_complete.is_set() and self.grpc_setup_complete.is_set():
            return json.dumps([ray.get_runtime_context().get_worker_id(), get_component_logger_file_path()])
        else:
            proxy_error = None
            if not self.http_setup_complete.is_set():
                try:
                    await done_set_http.pop()
                except Exception as e:
                    logger.exception(e)
                    proxy_error = e
            if not self.grpc_setup_complete.is_set():
                try:
                    await done_set_grpc.pop()
                except Exception as e:
                    logger.exception(e)
                    proxy_error = e
            raise proxy_error

    async def run_http_server(self):
        sock = socket.socket()
        if SOCKET_REUSE_PORT_ENABLED:
            set_socket_reuse_port(sock)
        try:
            sock.bind((self.host, self.port))
        except OSError:
            raise ValueError(f"Failed to bind Ray Serve HTTP proxy to '{self.host}:{self.port}'. Please make sure your http-host and http-port are specified correctly.")
        config = uvicorn.Config(self.wrapped_http_proxy, host=self.host, port=self.port, loop=_determine_target_loop(), root_path=self.root_path, lifespan='off', access_log=False, timeout_keep_alive=self.keep_alive_timeout_s)
        self._uvicorn_server = uvicorn.Server(config=config)
        self._uvicorn_server.install_signal_handlers = lambda: None
        logger.info(f'Starting HTTP server on node: {ray.get_runtime_context().get_node_id()} listening on port {self.port}')
        self.http_setup_complete.set()
        await self._uvicorn_server.serve(sockets=[sock])

    async def run_grpc_server(self):
        if not self.should_start_grpc_service():
            return self.grpc_setup_complete.set()
        grpc_server = create_serve_grpc_server(service_handler_factory=self.grpc_proxy.service_handler_factory)
        grpc_server.add_insecure_port(f'[::]:{self.grpc_port}')
        dummy_servicer = DummyServicer()
        add_RayServeAPIServiceServicer_to_server(dummy_servicer, grpc_server)
        for grpc_servicer_function in self.grpc_options.grpc_servicer_func_callable:
            grpc_servicer_function(dummy_servicer, grpc_server)
        await grpc_server.start()
        logger.info(f'Starting gRPC server on node: {ray.get_runtime_context().get_node_id()} listening on port {self.grpc_port}')
        self.grpc_setup_complete.set()
        await grpc_server.wait_for_termination()

    async def update_draining(self, draining: bool, _after: Optional[Any]=None):
        """Update the draining status of the HTTP and gRPC proxies.

        Unused `_after` argument is for scheduling: passing an ObjectRef
        allows delaying this call until after the `_after` call has returned.
        """
        self.http_proxy.update_draining(draining)
        if self.grpc_proxy:
            self.grpc_proxy.update_draining(draining)

    async def is_drained(self, _after: Optional[Any]=None):
        """Check whether both HTTP and gRPC proxies are drained or not.

        Unused `_after` argument is for scheduling: passing an ObjectRef
        allows delaying this call until after the `_after` call has returned.
        """
        return self.http_proxy.is_drained() and (self.grpc_proxy is None or self.grpc_proxy.is_drained())

    async def check_health(self):
        """No-op method to check on the health of the HTTP Proxy.
        Make sure the async event loop is not blocked.
        """
        logger.info('Received health check.', extra={'log_to_stderr': False})

    async def receive_asgi_messages(self, request_id: str) -> bytes:
        """Get ASGI messages for the provided `request_id`.

        After the proxy has stopped receiving messages for this `request_id`,
        this will always return immediately.
        """
        return pickle.dumps(await self.http_proxy.receive_asgi_messages(request_id))

    def _save_cpu_profile_data(self) -> str:
        """Saves CPU profiling data, if CPU profiling is enabled.

        Logs a warning if CPU profiling is disabled.
        """
        if self.cpu_profiler is not None:
            import marshal
            self.cpu_profiler.snapshot_stats()
            with open(self.cpu_profiler_log, 'wb') as f:
                marshal.dump(self.cpu_profiler.stats, f)
            logger.info(f'Saved CPU profile data to file "{self.cpu_profiler_log}"')
            return self.cpu_profiler_log
        else:
            logger.error('Attempted to save CPU profile data, but failed because no CPU profiler was running! Enable CPU profiling by enabling the RAY_SERVE_ENABLE_CPU_PROFILING env var.')

    async def _uvicorn_keep_alive(self) -> Optional[int]:
        """Get the keep alive timeout used for the running uvicorn server.

        Return the timeout_keep_alive config used on the uvicorn server if it's running.
        If the server is not running, return None.
        """
        if self._uvicorn_server:
            return self._uvicorn_server.config.timeout_keep_alive
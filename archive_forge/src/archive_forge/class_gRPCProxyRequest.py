import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Tuple, Union
import grpc
from starlette.types import Receive, Scope, Send
from ray.actor import ActorHandle
from ray.serve._private.common import StreamingHTTPRequest, gRPCRequest
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import DEFAULT
from ray.serve.grpc_util import RayServegRPCContext
class gRPCProxyRequest(ProxyRequest):
    """ProxyRequest implementation to wrap gRPC request protobuf and metadata."""

    def __init__(self, request_proto: Any, context: grpc._cython.cygrpc._ServicerContext, service_method: str, stream: bool):
        self.request = request_proto
        self.context = context
        self.service_method = service_method
        self.stream = stream
        self.app_name = ''
        self.request_id = None
        self.method_name = '__call__'
        self.multiplexed_model_id = DEFAULT.VALUE
        self.ray_serve_grpc_context = RayServegRPCContext(context)
        self.setup_variables()

    def setup_variables(self):
        if not self.is_route_request and (not self.is_health_request):
            service_method_split = self.service_method.split('/')
            self.request = pickle.dumps(self.request)
            self.method_name = service_method_split[-1]
            for key, value in self.context.invocation_metadata():
                if key == 'application':
                    self.app_name = value
                elif key == 'request_id':
                    self.request_id = value
                elif key == 'multiplexed_model_id':
                    self.multiplexed_model_id = value

    @property
    def request_type(self) -> str:
        return 'grpc'

    @property
    def method(self) -> str:
        return self.service_method

    @property
    def route_path(self) -> str:
        return self.app_name

    @property
    def is_route_request(self) -> bool:
        return self.service_method == '/ray.serve.RayServeAPIService/ListApplications'

    @property
    def is_health_request(self) -> bool:
        return self.service_method == '/ray.serve.RayServeAPIService/Healthz'

    @property
    def user_request(self) -> bytes:
        return self.request

    def send_request_id(self, request_id: str):
        self.ray_serve_grpc_context.set_trailing_metadata([('request_id', request_id)])

    def request_object(self, proxy_handle: ActorHandle) -> gRPCRequest:
        return gRPCRequest(grpc_user_request=self.user_request, grpc_proxy_handle=proxy_handle)
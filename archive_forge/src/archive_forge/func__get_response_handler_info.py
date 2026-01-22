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
def _get_response_handler_info(self, proxy_request: ProxyRequest) -> ResponseHandlerInfo:
    if proxy_request.is_route_request:
        if self._is_draining():
            return ResponseHandlerInfo(response_generator=self.draining_response(proxy_request), metadata=HandlerMetadata(), should_record_access_log=False, should_record_request_metrics=False, should_increment_ongoing_requests=False)
        else:
            return ResponseHandlerInfo(response_generator=self.routes_response(proxy_request), metadata=HandlerMetadata(application_name='', deployment_name='', route=proxy_request.route_path), should_record_access_log=False, should_record_request_metrics=True, should_increment_ongoing_requests=False)
    elif proxy_request.is_health_request:
        if self._is_draining():
            return ResponseHandlerInfo(response_generator=self.draining_response(proxy_request), metadata=HandlerMetadata(), should_record_access_log=False, should_record_request_metrics=False, should_increment_ongoing_requests=False)
        else:
            return ResponseHandlerInfo(response_generator=self.health_response(proxy_request), metadata=HandlerMetadata(application_name='', deployment_name='', route=proxy_request.route_path), should_record_access_log=False, should_record_request_metrics=True, should_increment_ongoing_requests=False)
    else:
        matched_route = None
        if self.protocol == RequestProtocol.HTTP:
            matched_route = self.proxy_router.match_route(proxy_request.route_path)
        elif self.protocol == RequestProtocol.GRPC:
            matched_route = self.proxy_router.get_handle_for_endpoint(proxy_request.route_path)
        if matched_route is None:
            return ResponseHandlerInfo(response_generator=self.not_found(proxy_request), metadata=HandlerMetadata(application_name='', deployment_name='', route=proxy_request.route_path), should_record_access_log=True, should_record_request_metrics=True, should_increment_ongoing_requests=False)
        else:
            route_prefix, handle, app_is_cross_language = matched_route
            route_path = proxy_request.route_path
            if route_prefix != '/' and self.protocol == RequestProtocol.HTTP:
                assert not route_prefix.endswith('/')
                proxy_request.set_path(route_path.replace(route_prefix, '', 1))
                proxy_request.set_root_path(proxy_request.root_path + route_prefix)
            handle, request_id = self.setup_request_context_and_handle(app_name=handle.deployment_id.app, handle=handle, route_path=route_path, proxy_request=proxy_request)
            response_generator = self.send_request_to_replica(request_id=request_id, handle=handle, proxy_request=proxy_request, app_is_cross_language=app_is_cross_language)
            return ResponseHandlerInfo(response_generator=response_generator, metadata=HandlerMetadata(application_name=handle.deployment_id.app, deployment_name=handle.deployment_id.name, route=route_path), should_record_access_log=True, should_record_request_metrics=True, should_increment_ongoing_requests=True)
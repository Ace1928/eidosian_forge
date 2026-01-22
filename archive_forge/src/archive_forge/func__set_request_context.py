import contextvars
import logging
from dataclasses import dataclass
from typing import Callable, Optional
import ray
from ray.exceptions import RayActorError
from ray.serve._private.client import ServeControllerClient
from ray.serve._private.common import ReplicaTag
from ray.serve._private.constants import SERVE_CONTROLLER_NAME, SERVE_NAMESPACE
from ray.serve.exceptions import RayServeException
from ray.serve.grpc_util import RayServegRPCContext
from ray.util.annotations import DeveloperAPI
def _set_request_context(route: str='', request_id: str='', app_name: str='', multiplexed_model_id: str=''):
    """Set the request context. If the value is not set,
    the current context value will be used."""
    current_request_context = _serve_request_context.get()
    _serve_request_context.set(_RequestContext(route=route or current_request_context.route, request_id=request_id or current_request_context.request_id, app_name=app_name or current_request_context.app_name, multiplexed_model_id=multiplexed_model_id or current_request_context.multiplexed_model_id))
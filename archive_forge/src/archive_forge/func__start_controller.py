import inspect
import logging
from types import FunctionType
from typing import Any, Dict, Tuple, Union
import ray
from ray._private.pydantic_compat import is_subclass_of_base_model
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.usage import usage_lib
from ray.actor import ActorHandle
from ray.serve._private.client import ServeControllerClient
from ray.serve._private.constants import (
from ray.serve._private.controller import ServeController
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.context import _get_global_client, _set_global_client
from ray.serve.deployment import Application, Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.schema import LoggingConfig
def _start_controller(http_options: Union[None, dict, HTTPOptions]=None, grpc_options: Union[None, dict, gRPCOptions]=None, global_logging_config: Union[None, dict, LoggingConfig]=None, **kwargs) -> Tuple[ActorHandle, str]:
    """Start Ray Serve controller.

    The function makes sure controller is ready to start deploying apps
    after it returns.

    Parameters are same as ray.serve._private.api.serve_start().

    Returns: A tuple with controller actor handle and controller name.
    """
    ray._private.worker.global_worker._filter_logs_by_job = False
    if not ray.is_initialized():
        ray.init(namespace=SERVE_NAMESPACE)
    controller_actor_options = {'num_cpus': 0, 'name': SERVE_CONTROLLER_NAME, 'lifetime': 'detached', 'max_restarts': -1, 'max_task_retries': -1, 'resources': {HEAD_NODE_RESOURCE_NAME: 0.001}, 'namespace': SERVE_NAMESPACE, 'max_concurrency': CONTROLLER_MAX_CONCURRENCY}
    http_deprecated_args = ['http_host', 'http_port', 'http_middlewares']
    for key in http_deprecated_args:
        if key in kwargs:
            raise ValueError(f'{key} is deprecated, please use serve.start(http_options={{"{key}": {kwargs[key]}}}) instead.')
    if isinstance(http_options, dict):
        http_options = HTTPOptions.parse_obj(http_options)
    if http_options is None:
        http_options = HTTPOptions()
    if isinstance(grpc_options, dict):
        grpc_options = gRPCOptions(**grpc_options)
    if global_logging_config is None:
        global_logging_config = LoggingConfig()
    elif isinstance(global_logging_config, dict):
        global_logging_config = LoggingConfig(**global_logging_config)
    controller = ServeController.options(**controller_actor_options).remote(SERVE_CONTROLLER_NAME, http_config=http_options, grpc_options=grpc_options, global_logging_config=global_logging_config)
    proxy_handles = ray.get(controller.get_proxies.remote())
    if len(proxy_handles) > 0:
        try:
            ray.get([handle.ready.remote() for handle in proxy_handles.values()], timeout=HTTP_PROXY_TIMEOUT)
        except ray.exceptions.GetTimeoutError:
            raise TimeoutError(f'HTTP proxies not available after {HTTP_PROXY_TIMEOUT}s.')
    return (controller, SERVE_CONTROLLER_NAME)
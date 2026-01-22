import asyncio
import concurrent.futures
import threading
import warnings
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple, Union
import ray
from ray import serve
from ray._raylet import GcsClient, ObjectRefGenerator
from ray.serve._private.common import DeploymentID, RequestProtocol
from ray.serve._private.default_impl import create_cluster_node_info_cache
from ray.serve._private.router import RequestMetadata, Router
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.util import metrics
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
class _DeploymentHandleBase:

    def __init__(self, deployment_name: str, app_name: str, *, sync: bool, handle_options: Optional[_HandleOptions]=None, _router: Optional[Router]=None, _request_counter: Optional[metrics.Counter]=None, _recorded_telemetry: bool=False):
        self.deployment_id = DeploymentID(deployment_name, app_name)
        self.handle_options = handle_options or _HandleOptions()
        self._recorded_telemetry = _recorded_telemetry
        self._sync = sync
        self.request_counter = _request_counter or self._create_request_counter(app_name, deployment_name)
        self._router: Optional[Router] = _router

    def _record_telemetry_if_needed(self):
        if not self._recorded_telemetry and self.handle_options._request_protocol == RequestProtocol.UNDEFINED:
            if self.__class__ == DeploymentHandle:
                ServeUsageTag.DEPLOYMENT_HANDLE_API_USED.record('1')
            elif self.__class__ == RayServeHandle:
                ServeUsageTag.RAY_SERVE_HANDLE_API_USED.record('1')
            else:
                ServeUsageTag.RAY_SERVE_SYNC_HANDLE_API_USED.record('1')
            self._recorded_telemetry = True

    def _set_request_protocol(self, request_protocol: RequestProtocol):
        self.handle_options = self.handle_options.copy_and_update(_request_protocol=request_protocol)

    def _get_or_create_router(self) -> Union[Router, asyncio.AbstractEventLoop]:
        if self._router is None:
            node_id = ray.get_runtime_context().get_node_id()
            try:
                cluster_node_info_cache = create_cluster_node_info_cache(GcsClient(address=ray.get_runtime_context().gcs_address))
                cluster_node_info_cache.update()
                availability_zone = cluster_node_info_cache.get_node_az(node_id)
            except Exception:
                availability_zone = None
            self._router = Router(serve.context._get_global_client()._controller, self.deployment_id, node_id, get_current_actor_id(), availability_zone, event_loop=_create_or_get_global_asyncio_event_loop_in_thread(), _prefer_local_node_routing=self.handle_options._prefer_local_routing, _router_cls=self.handle_options._router_cls)
        return (self._router, self._router._event_loop)

    @staticmethod
    def _gen_handle_tag(app_name: str, deployment_name: str, handle_id: str):
        if app_name:
            return f'{app_name}#{deployment_name}#{handle_id}'
        else:
            return f'{deployment_name}#{handle_id}'

    @classmethod
    def _create_request_counter(cls, app_name, deployment_name):
        return metrics.Counter('serve_handle_request_counter', description='The number of handle.remote() calls that have been made on this handle.', tag_keys=('handle', 'deployment', 'route', 'application')).set_default_tags({'handle': cls._gen_handle_tag(app_name, deployment_name, handle_id=get_random_letters()), 'deployment': deployment_name, 'application': app_name})

    @property
    def deployment_name(self) -> str:
        return self.deployment_id.name

    @property
    def app_name(self) -> str:
        return self.deployment_id.app

    def _options(self, *, method_name: Union[str, DEFAULT]=DEFAULT.VALUE, multiplexed_model_id: Union[str, DEFAULT]=DEFAULT.VALUE, stream: Union[bool, DEFAULT]=DEFAULT.VALUE, use_new_handle_api: Union[bool, DEFAULT]=DEFAULT.VALUE, _prefer_local_routing: Union[bool, DEFAULT]=DEFAULT.VALUE, _router_cls: Union[str, DEFAULT]=DEFAULT.VALUE):
        new_handle_options = self.handle_options.copy_and_update(method_name=method_name, multiplexed_model_id=multiplexed_model_id, stream=stream, _prefer_local_routing=_prefer_local_routing, _router_cls=_router_cls)
        if self._router is None and _router_cls == DEFAULT.VALUE and (_prefer_local_routing == DEFAULT.VALUE):
            self._get_or_create_router()
        if use_new_handle_api is True:
            cls = DeploymentHandle
        elif use_new_handle_api is False:
            if self._sync:
                cls = RayServeSyncHandle
            else:
                cls = RayServeHandle
        else:
            cls = self.__class__
        return cls(self.deployment_name, self.app_name, handle_options=new_handle_options, sync=self._sync, _router=None if _router_cls != DEFAULT.VALUE else self._router, _request_counter=self.request_counter, _recorded_telemetry=self._recorded_telemetry)

    def _remote(self, args: Tuple[Any], kwargs: Dict[str, Any]) -> concurrent.futures.Future:
        if not self.__class__ == DeploymentHandle:
            warnings.warn('`DeploymentHandle` is now the default handle API. You can continue using the existing `RayServeHandle` and `RayServeSyncHandle` APIs by calling `handle.options(use_new_handle_api=False)` or setting the global environment variable `RAY_SERVE_ENABLE_NEW_HANDLE_API=0`, but support for these will be removed in a future release. See https://docs.ray.io/en/latest/serve/model_composition.html for more details.')
        self._record_telemetry_if_needed()
        _request_context = ray.serve.context._serve_request_context.get()
        request_metadata = RequestMetadata(_request_context.request_id, self.deployment_name, call_method=self.handle_options.method_name, route=_request_context.route, app_name=self.app_name, multiplexed_model_id=self.handle_options.multiplexed_model_id, is_streaming=self.handle_options.stream, _request_protocol=self.handle_options._request_protocol, grpc_context=_request_context.grpc_context)
        self.request_counter.inc(tags={'route': _request_context.route, 'application': _request_context.app_name})
        router, event_loop = self._get_or_create_router()
        return asyncio.run_coroutine_threadsafe(router.assign_request(request_metadata, *args, **kwargs), loop=event_loop)

    def __getattr__(self, name):
        return self.options(method_name=name)

    def shutdown(self):
        if self._router:
            self._router.shutdown()

    def __repr__(self):
        return f"{self.__class__.__name__}(deployment='{self.deployment_name}')"

    @classmethod
    def _deserialize(cls, kwargs):
        """Required for this class's __reduce__ method to be picklable."""
        return cls(**kwargs)

    def __reduce__(self):
        serialized_constructor_args = {'deployment_name': self.deployment_name, 'app_name': self.app_name, 'handle_options': self.handle_options, 'sync': self._sync}
        return (self.__class__._deserialize, (serialized_constructor_args,))
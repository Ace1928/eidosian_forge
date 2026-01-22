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
@Deprecated(message='This API has been replaced by `ray.serve.handle.DeploymentHandle`.')
class RayServeHandle(_DeploymentHandleBase):
    """A handle used to make requests from one deployment to another.

    This is used to compose multiple deployments into a single application. After
    building the application, this handle is substituted at runtime for deployments
    passed as arguments via `.bind()`.

    Example:

    .. code-block:: python

        import ray
        from ray import serve
        from ray.serve.handle import RayServeHandle, RayServeSyncHandle

        @serve.deployment
        class Downstream:
            def __init__(self, message: str):
                self._message = message

            def __call__(self, name: str) -> str:
                return self._message + name

        @serve.deployment
        class Ingress:
            def __init__(self, handle: RayServeHandle):
                self._handle = handle

            async def __call__(self, name: str) -> str:
                obj_ref: ray.ObjectRef = await self._handle.remote(name)
                return await obj_ref

        app = Ingress.bind(Downstream.bind("Hello "))
        handle: RayServeSyncHandle = serve.run(app)

        # Prints "Hello Mr. Magoo"
        print(ray.get(handle.remote("Mr. Magoo")))

    """

    def options(self, *, method_name: Union[str, DEFAULT]=DEFAULT.VALUE, multiplexed_model_id: Union[str, DEFAULT]=DEFAULT.VALUE, stream: Union[bool, DEFAULT]=DEFAULT.VALUE, use_new_handle_api: Union[bool, DEFAULT]=DEFAULT.VALUE, _prefer_local_routing: Union[bool, DEFAULT]=DEFAULT.VALUE, _router_cls: Union[str, DEFAULT]=DEFAULT.VALUE) -> 'RayServeHandle':
        """Set options for this handle and return an updated copy of it.

        Example:

        .. code-block:: python

            # The following two lines are equivalent:
            obj_ref = await handle.other_method.remote(*args)
            obj_ref = await handle.options(method_name="other_method").remote(*args)
            obj_ref = await handle.options(
                multiplexed_model_id="model:v1").remote(*args)
        """
        return self._options(method_name=method_name, multiplexed_model_id=multiplexed_model_id, stream=stream, _prefer_local_routing=_prefer_local_routing, use_new_handle_api=use_new_handle_api, _router_cls=_router_cls)

    def remote(self, *args, **kwargs) -> asyncio.Task:
        """Issue an asynchronous request to the __call__ method of the deployment.

        Returns an `asyncio.Task` whose underlying result is a Ray ObjectRef that
        points to the final result of the request.

        The final result can be retrieved by awaiting the ObjectRef.

        Example:

        .. code-block:: python

            obj_ref = await handle.remote(*args)
            result = await obj_ref

        """
        future = self._remote(args, kwargs)

        async def await_future():
            return await asyncio.wrap_future(future)
        task = asyncio.ensure_future(await_future())
        task._ray_serve_object_ref_future = future
        return task
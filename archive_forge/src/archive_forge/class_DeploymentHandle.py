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
@PublicAPI(stability='beta')
class DeploymentHandle(_DeploymentHandleBase):
    """A handle used to make requests to a deployment at runtime.

    This is primarily used to compose multiple deployments within a single application.
    It can also be used to make calls to the ingress deployment of an application (e.g.,
    for programmatic testing).

    Example:


    .. code-block:: python

        import ray
        from ray import serve
        from ray.serve.handle import DeploymentHandle, DeploymentResponse

        @serve.deployment
        class Downstream:
            def say_hi(self, message: str):
                return f"Hello {message}!"
                self._message = message

        @serve.deployment
        class Ingress:
            def __init__(self, handle: DeploymentHandle):
                self._downstream_handle = handle

            async def __call__(self, name: str) -> str:
                response = self._handle.say_hi.remote(name)
                return await response

        app = Ingress.bind(Downstream.bind())
        handle: DeploymentHandle = serve.run(app)
        response = handle.remote("world")
        assert response.result() == "Hello world!"
    """

    def options(self, *, method_name: Union[str, DEFAULT]=DEFAULT.VALUE, multiplexed_model_id: Union[str, DEFAULT]=DEFAULT.VALUE, stream: Union[bool, DEFAULT]=DEFAULT.VALUE, use_new_handle_api: Union[bool, DEFAULT]=DEFAULT.VALUE, _prefer_local_routing: Union[bool, DEFAULT]=DEFAULT.VALUE, _router_cls: Union[str, DEFAULT]=DEFAULT.VALUE) -> 'DeploymentHandle':
        """Set options for this handle and return an updated copy of it.

        Example:

        .. code-block:: python

            response: DeploymentResponse = handle.options(
                method_name="other_method",
                multiplexed_model_id="model:v1",
            ).remote()
        """
        return self._options(method_name=method_name, multiplexed_model_id=multiplexed_model_id, stream=stream, use_new_handle_api=use_new_handle_api, _prefer_local_routing=_prefer_local_routing, _router_cls=_router_cls)

    def remote(self, *args, **kwargs) -> Union[DeploymentResponse, DeploymentResponseGenerator]:
        """Issue a remote call to a method of the deployment.

        By default, the result is a `DeploymentResponse` that can be awaited to fetch
        the result of the call or passed to another `.remote()` call to compose multiple
        deployments.

        If `handle.options(stream=True)` is set and a generator method is called, this
        returns a `DeploymentResponseGenerator` instead.

        Example:

        .. code-block:: python

            # Fetch the result directly.
            response = handle.remote()
            result = await response

            # Pass the result to another handle call.
            composed_response = handle2.remote(handle1.remote())
            composed_result = await composed_response

        Args:
            *args: Positional arguments to be serialized and passed to the
                remote method call.
            **kwargs: Keyword arguments to be serialized and passed to the
                remote method call.
        """
        future = self._remote(args, kwargs)
        if self.handle_options.stream:
            response_cls = DeploymentResponseGenerator
        else:
            response_cls = DeploymentResponse
        return response_cls(future)
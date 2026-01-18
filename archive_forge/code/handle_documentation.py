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
Issue a remote call to a method of the deployment.

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
        
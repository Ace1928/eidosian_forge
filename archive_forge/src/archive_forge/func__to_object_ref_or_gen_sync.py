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
def _to_object_ref_or_gen_sync(self, _record_telemetry: bool=True, _allow_running_in_asyncio_loop: bool=False) -> Union[ray.ObjectRef, ObjectRefGenerator]:
    if not _allow_running_in_asyncio_loop and is_running_in_asyncio_loop():
        raise RuntimeError('Sync methods should not be called from within an `asyncio` event loop. Use `await response` or `await response._to_object_ref()` instead.')
    if _record_telemetry:
        ServeUsageTag.DEPLOYMENT_HANDLE_TO_OBJECT_REF_API_USED.record('1')
    return self._object_ref_future.result()
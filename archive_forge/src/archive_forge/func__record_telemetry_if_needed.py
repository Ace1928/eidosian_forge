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
def _record_telemetry_if_needed(self):
    if not self._recorded_telemetry and self.handle_options._request_protocol == RequestProtocol.UNDEFINED:
        if self.__class__ == DeploymentHandle:
            ServeUsageTag.DEPLOYMENT_HANDLE_API_USED.record('1')
        elif self.__class__ == RayServeHandle:
            ServeUsageTag.RAY_SERVE_HANDLE_API_USED.record('1')
        else:
            ServeUsageTag.RAY_SERVE_SYNC_HANDLE_API_USED.record('1')
        self._recorded_telemetry = True
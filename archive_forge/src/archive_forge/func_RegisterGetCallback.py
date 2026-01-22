import math
import logging
import queue
import threading
import warnings
import grpc
from collections import OrderedDict
from typing import Any, Callable, Dict, TYPE_CHECKING, Optional, Union
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.client.common import (
from ray.util.debug import log_once
def RegisterGetCallback(self, request: ray_client_pb2.GetRequest, callback: ResponseCallable) -> None:
    if len(request.ids) != 1:
        raise ValueError(f'RegisterGetCallback() must have exactly 1 Object ID. Actual: {request}')
    datareq = ray_client_pb2.DataRequest(get=request)
    collector = ChunkCollector(callback=callback, request=datareq)
    self._async_send(datareq, collector)
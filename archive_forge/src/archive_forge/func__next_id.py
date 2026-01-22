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
def _next_id(self) -> int:
    assert self.lock.locked()
    self._req_id += 1
    if self._req_id > INT32_MAX:
        self._req_id = 1
    assert self._req_id != 0
    return self._req_id
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
def _blocking_send(self, req: ray_client_pb2.DataRequest) -> ray_client_pb2.DataResponse:
    with self.lock:
        self._check_shutdown()
        req_id = self._next_id()
        req.req_id = req_id
        self.request_queue.put(req)
        self.outstanding_requests[req_id] = req
        self.cv.wait_for(lambda: req_id in self.ready_data or self._in_shutdown)
        self._check_shutdown()
        data = self.ready_data[req_id]
        del self.ready_data[req_id]
        del self.outstanding_requests[req_id]
        self._acknowledge(req_id)
    return data
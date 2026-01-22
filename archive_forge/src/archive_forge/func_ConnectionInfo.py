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
def ConnectionInfo(self, context=None) -> ray_client_pb2.ConnectionInfoResponse:
    datareq = ray_client_pb2.DataRequest(connection_info=ray_client_pb2.ConnectionInfoRequest())
    resp = self._blocking_send(datareq)
    return resp.connection_info
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
def _can_reconnect(self, e: grpc.RpcError) -> bool:
    """
        Processes RPC errors that occur while reading from data stream.
        Returns True if the error can be recovered from, False otherwise.
        """
    if not self.client_worker._can_reconnect(e):
        logger.error('Unrecoverable error in data channel.')
        logger.debug(e)
        return False
    logger.debug('Recoverable error in data channel.')
    logger.debug(e)
    return True
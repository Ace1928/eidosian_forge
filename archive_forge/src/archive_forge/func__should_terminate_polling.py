import asyncio
from collections import deque
import logging
import random
from typing import Tuple, List
import grpc
from ray._private.utils import get_or_create_event_loop
import ray._private.gcs_utils as gcs_utils
import ray._private.logging_utils as logging_utils
from ray.core.generated.gcs_pb2 import ErrorTableData
from ray.core.generated import dependency_pb2
from ray.core.generated import gcs_service_pb2_grpc
from ray.core.generated import gcs_service_pb2
from ray.core.generated import common_pb2
from ray.core.generated import pubsub_pb2
@staticmethod
def _should_terminate_polling(e: grpc.RpcError) -> None:
    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        return True
    if e.code() == grpc.StatusCode.UNAVAILABLE:
        return True
    return False
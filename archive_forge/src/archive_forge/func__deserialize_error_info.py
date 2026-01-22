import io
import logging
import threading
import traceback
from typing import Any
import google.protobuf.message
import ray._private.utils
import ray.cloudpickle as pickle
from ray._private import ray_constants
from ray._raylet import (
from ray.core.generated.common_pb2 import ErrorType, RayErrorInfo
from ray.exceptions import (
from ray.util import serialization_addons
from ray.util import inspect_serializability
def _deserialize_error_info(self, data, metadata_fields):
    assert data
    pb_bytes = self._deserialize_msgpack_data(data, metadata_fields)
    assert pb_bytes
    ray_error_info = RayErrorInfo()
    ray_error_info.ParseFromString(pb_bytes)
    return ray_error_info
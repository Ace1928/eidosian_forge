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
def _deserialize_msgpack_data(self, data, metadata_fields):
    msgpack_data, pickle5_data = split_buffer(data)
    if metadata_fields[0] == ray_constants.OBJECT_METADATA_TYPE_PYTHON:
        python_objects = self._deserialize_pickle5_data(pickle5_data)
    else:
        python_objects = []
    try:

        def _python_deserializer(index):
            return python_objects[index]
        obj = MessagePackSerializer.loads(msgpack_data, _python_deserializer)
    except Exception:
        raise DeserializationError()
    return obj
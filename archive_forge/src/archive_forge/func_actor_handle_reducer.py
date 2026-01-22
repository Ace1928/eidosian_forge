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
def actor_handle_reducer(obj):
    ray._private.worker.global_worker.check_connected()
    serialized, actor_handle_id = obj._serialization_helper()
    self.add_contained_object_ref(actor_handle_id)
    return (_actor_handle_deserializer, (serialized,))
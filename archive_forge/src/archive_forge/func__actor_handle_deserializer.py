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
def _actor_handle_deserializer(serialized_obj):
    context = ray._private.worker.global_worker.get_serialization_context()
    outer_id = context.get_outer_object_ref()
    return ray.actor.ActorHandle._deserialization_helper(serialized_obj, outer_id)
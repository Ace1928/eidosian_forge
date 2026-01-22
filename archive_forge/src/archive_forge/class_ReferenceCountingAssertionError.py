import os
from traceback import format_exception
from typing import Optional, Union
import colorama
import ray._private.ray_constants as ray_constants
import ray.cloudpickle as pickle
from ray._raylet import ActorID, TaskID, WorkerID
from ray.core.generated.common_pb2 import (
from ray.util.annotations import DeveloperAPI, PublicAPI
import setproctitle
@DeveloperAPI
class ReferenceCountingAssertionError(ObjectLostError, AssertionError):
    """Indicates that an object has been deleted while there was still a
    reference to it.

    Args:
        object_ref_hex: Hex ID of the object.
    """

    def __str__(self):
        return self._base_str() + '\n\n' + 'The object has already been deleted by the reference counting protocol. This should not happen.'
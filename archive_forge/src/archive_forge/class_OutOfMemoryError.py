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
@PublicAPI
class OutOfMemoryError(RayError):
    """Indicates that the node is running out of memory and is close to full.

    This is raised if the node is low on memory and tasks or actors are being
    evicted to free up memory.
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
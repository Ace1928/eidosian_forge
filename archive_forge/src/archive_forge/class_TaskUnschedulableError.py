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
class TaskUnschedulableError(RayError):
    """Raised when the task cannot be scheduled.

    One example is that the node specified through
    NodeAffinitySchedulingStrategy is dead.
    """

    def __init__(self, error_message: str):
        self.error_message = error_message

    def __str__(self):
        return f'The task is not schedulable: {self.error_message}'
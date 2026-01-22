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
class ObjectLostError(RayError):
    """Indicates that the object is lost from distributed memory, due to
    node failure or system error.

    Args:
        object_ref_hex: Hex ID of the object.
    """

    def __init__(self, object_ref_hex, owner_address, call_site):
        self.object_ref_hex = object_ref_hex
        self.owner_address = owner_address
        self.call_site = call_site.replace(ray_constants.CALL_STACK_LINE_DELIMITER, '\n  ')

    def _base_str(self):
        msg = f'Failed to retrieve object {self.object_ref_hex}. '
        if self.call_site:
            msg += f'The ObjectRef was created at: {self.call_site}'
        else:
            msg += 'To see information about where this ObjectRef was created in Python, set the environment variable RAY_record_ref_creation_sites=1 during `ray start` and `ray.init()`.'
        return msg

    def __str__(self):
        return self._base_str() + '\n\n' + f'All copies of {self.object_ref_hex} have been lost due to node failure. Check cluster logs (`/tmp/ray/session_latest/logs`) for more information about the failure.'
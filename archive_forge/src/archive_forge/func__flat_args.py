from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
@classmethod
def _flat_args(cls, args: Iterable):
    """
        Check if the arguments list is flat and subscribe to all `DeferredExecution` objects.

        Parameters
        ----------
        args : Iterable

        Returns
        -------
        bool
        """
    flat = True
    for arg in args:
        if isinstance(arg, DeferredExecution):
            flat = False
            arg.subscribe()
        elif isinstance(arg, ListOrTuple):
            flat = False
            cls._flat_args(arg)
    return flat
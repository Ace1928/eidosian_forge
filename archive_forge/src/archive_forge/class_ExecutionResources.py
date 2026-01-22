import os
from dataclasses import dataclass, field
from typing import List, Optional, Union
from .common import NodeIdStr
from ray.data._internal.execution.util import memory_string
from ray.util.annotations import DeveloperAPI
@dataclass
class ExecutionResources:
    """Specifies resources usage or resource limits for execution.

    The value `None` represents unknown resource usage or an unspecified limit.
    """
    cpu: Optional[float] = None
    gpu: Optional[float] = None
    object_store_memory: Optional[int] = None

    def object_store_memory_str(self) -> str:
        """Returns a human-readable string for the object store memory field."""
        if self.object_store_memory is None:
            return 'None'
        else:
            return memory_string(self.object_store_memory)

    def add(self, other: 'ExecutionResources') -> 'ExecutionResources':
        """Adds execution resources.

        Returns:
            A new ExecutionResource object with summed resources.
        """
        total = ExecutionResources()
        if self.cpu is not None or other.cpu is not None:
            total.cpu = (self.cpu or 0.0) + (other.cpu or 0.0)
        if self.gpu is not None or other.gpu is not None:
            total.gpu = (self.gpu or 0.0) + (other.gpu or 0.0)
        if self.object_store_memory is not None or other.object_store_memory is not None:
            total.object_store_memory = (self.object_store_memory or 0.0) + (other.object_store_memory or 0.0)
        return total

    def satisfies_limit(self, limit: 'ExecutionResources') -> bool:
        """Return if this resource struct meets the specified limits.

        Note that None for a field means no limit.
        """
        if self.cpu is not None and limit.cpu is not None and (self.cpu > limit.cpu):
            return False
        if self.gpu is not None and limit.gpu is not None and (self.gpu > limit.gpu):
            return False
        if self.object_store_memory is not None and limit.object_store_memory is not None and (self.object_store_memory > limit.object_store_memory):
            return False
        return True

    def scale(self, f: float) -> 'ExecutionResources':
        """Return copy with all set values scaled by `f`."""
        return ExecutionResources(cpu=self.cpu * f if self.cpu is not None else None, gpu=self.gpu * f if self.gpu is not None else None, object_store_memory=self.object_store_memory * f if self.object_store_memory is not None else None)
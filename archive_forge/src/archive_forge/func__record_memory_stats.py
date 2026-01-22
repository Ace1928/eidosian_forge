from collections import defaultdict
from itertools import chain
import pickle
from typing import (
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torch.utils._python_dispatch import TorchDispatchMode
@no_type_check
def _record_memory_stats(self, fn_name: str) -> None:
    """
        Record current memory allocated, current memory active and current memory reserved.

        The memory stats dict is indexed with ``self._op_index``.
        """
    memory_allocated: float = torch.cuda.memory_allocated() / BYTES_PER_MB
    memory_reserved: float = torch.cuda.memory_reserved() / BYTES_PER_MB
    memory_active: float = torch.cuda.memory_stats().get('active_bytes.all.current', 0) / BYTES_PER_MB
    self.memories_allocated[self._op_index] = (fn_name, memory_allocated)
    self.memories_reserved[self._op_index] = (fn_name, memory_reserved)
    self.memories_active[self._op_index] = (fn_name, memory_active)
    self._op_index += 1
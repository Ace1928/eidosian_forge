from collections import defaultdict
from itertools import chain
import pickle
from typing import (
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torch.utils._python_dispatch import TorchDispatchMode
class MemoryProfileDispatchMode(TorchDispatchMode):
    """Run in ``TorchDispatchMode`` to get memory stats at operator level."""

    def __init__(self, memory_tracker) -> None:
        self.memory_tracker = memory_tracker

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        rs = func(*args, **kwargs)
        if func == torch.ops.aten.detach.default:
            return rs
        func_name: str = self.memory_tracker._cur_module_name + '.' + func.__name__ + '_' + str(self.memory_tracker._operator_names[func.__name__])
        self.memory_tracker._operator_names[func.__name__] = self.memory_tracker._operator_names[func.__name__] + 1
        self.memory_tracker._record_memory_stats(func_name)
        return rs
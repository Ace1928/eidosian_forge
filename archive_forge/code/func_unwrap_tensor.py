from __future__ import annotations
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
def unwrap_tensor(v):
    if isinstance(v, WrappedTensor):
        return v.tensor
    if isinstance(v, debugger_constexpr):
        return v.value
    return v
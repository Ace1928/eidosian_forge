import contextlib
import ctypes
import importlib
import inspect
import sys
import types
from typing import Any, Callable, Dict, List, Type, Union
import torch._C
import torch.utils._pytree as pytree
from torch import _utils_internal
from torch._functorch.pyfunctorch import dispatch_functorch
class _PyOpNamespace(_OpNamespace):

    def __init__(self, name, ops):
        super().__init__(name)
        self._ops = ops

    def __getattr__(self, name):
        op = self._ops.get(name, None)
        if op is None:
            raise AttributeError(f"'_PyOpNamespace' '{self.name}' object has no attribute '{name}'")
        setattr(self, name, op)
        return op
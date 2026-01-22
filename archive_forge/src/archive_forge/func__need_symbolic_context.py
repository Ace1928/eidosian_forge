from __future__ import annotations
import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import (
@_beartype.beartype
def _need_symbolic_context(symbolic_fn: Callable) -> bool:
    """Checks if the first argument to symbolic_fn is annotated as type `torch.onnx.SymbolicContext`."""
    params = tuple(inspect.signature(symbolic_fn).parameters.values())
    if not params:
        return False
    first_param_name = params[0].name
    type_hints = typing.get_type_hints(symbolic_fn)
    if first_param_name not in type_hints:
        return False
    param_type = type_hints[first_param_name]
    return issubclass(param_type, _exporter_states.SymbolicContext)
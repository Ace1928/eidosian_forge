from __future__ import annotations
import functools
import inspect
import sys
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _type_utils, errors
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils
from torch.types import Number
@_beartype.beartype
def _avgpool_helper(tuple_fn: Callable[[Any], Sequence[int]], padding: Union[int, Sequence[int]], kernel_size, stride, divisor_override, name) -> Tuple[int, ...]:
    if divisor_override and divisor_override.node().kind() != 'prim::Constant':
        _unimplemented(name, 'divisor_override')
    return tuple(tuple_fn(padding))
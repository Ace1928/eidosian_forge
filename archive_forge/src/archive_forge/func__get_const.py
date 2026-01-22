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
def _get_const(value, desc, arg_name):
    if not _is_constant(value):
        raise errors.SymbolicValueError(f"ONNX symbolic expected a constant value of the '{arg_name}' argument, got '{value}'", value)
    return _parse_arg(value, desc)
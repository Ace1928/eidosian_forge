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
def _onnx_unsupported(op_name: str, value: Optional[_C.Value]=None) -> NoReturn:
    message = f'Unsupported: ONNX export of operator {op_name}. Please feel free to request support or submit a pull request on PyTorch GitHub: {_constants.PYTORCH_GITHUB_ISSUES_URL}'
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(message, value)
    raise errors.OnnxExporterError(message)
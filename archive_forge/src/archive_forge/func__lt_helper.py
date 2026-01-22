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
def _lt_helper(g: jit_utils.GraphContext, input, other):
    if g.opset <= 8:
        from torch.onnx.symbolic_opset8 import lt as _lt8
        return _lt8(g, input, other)
    else:
        from torch.onnx.symbolic_opset9 import lt as _lt9
        return _lt9(g, input, other)
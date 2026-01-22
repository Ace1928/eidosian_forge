from __future__ import annotations
import functools
import sys
import warnings
from typing import Optional, Sequence
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::__getitem_')
@_beartype.beartype
def __getitem_(g: jit_utils.GraphContext, self, i):
    if symbolic_helper._is_tensor_list(self):
        return g.op('SequenceAt', self, i)
    else:
        from torch.onnx.symbolic_opset9 import __getitem_ as getitem
        return getitem(g, self, i)
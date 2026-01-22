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
@_onnx_symbolic('aten::_unique2')
@symbolic_helper.parse_args('v', 'i', 'i', 'i')
@_beartype.beartype
def _unique2(g: jit_utils.GraphContext, self, sorted, return_inverse, return_counts):
    u, indices, inverse_indices, counts = g.op('Unique', self, sorted_i=sorted, outputs=4)
    return (u, inverse_indices, counts)
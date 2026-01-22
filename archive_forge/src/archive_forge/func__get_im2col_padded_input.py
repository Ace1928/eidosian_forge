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
@_beartype.beartype
def _get_im2col_padded_input(g: jit_utils.GraphContext, input, padding_h, padding_w):
    pad = g.op('Constant', value_t=torch.LongTensor([0, 0, padding_h, padding_w] * 2))
    return g.op('Pad', input, pad)
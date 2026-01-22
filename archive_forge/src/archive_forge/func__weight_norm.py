from __future__ import annotations
import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number
@_onnx_symbolic('aten::_weight_norm')
@symbolic_helper.parse_args('v', 'v', 'i')
@_beartype.beartype
def _weight_norm(g: jit_utils.GraphContext, weight_v, weight_g, dim):
    rank = symbolic_helper._get_tensor_rank(weight_v)
    if rank is not None:
        axes = list(range(rank))
        if dim is not None:
            if dim < -1:
                dim += rank
            if dim != -1:
                axes.remove(dim)
        norm_v = norm(g, weight_v, 2, axes, 1)
        div = g.op('Div', weight_v, norm_v)
        return g.op('Mul', div, weight_g)
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at('_weight_norm', weight_v, weight_g, dim_i=dim)
    raise errors.SymbolicValueError('Unsupported: ONNX export of _weight_norm for tensor of unknown rank.', weight_v)
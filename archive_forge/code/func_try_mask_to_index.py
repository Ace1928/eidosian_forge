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
@_beartype.beartype
def try_mask_to_index(index):
    if not symbolic_helper._is_none(index) and (_type_utils.JitScalarType.from_value(index, _type_utils.JitScalarType.UNDEFINED) == _type_utils.JitScalarType.UINT8 or symbolic_helper._is_bool(index)):
        if g.opset < 9:
            raise errors.SymbolicValueError('Exporting masked indices are only supported after ONNX opset 9.', self)
        warnings.warn('Exporting aten::index operator with indices of type Byte. Only 1-D indices are supported. In any other case, this will produce an incorrect ONNX graph.')
        index = symbolic_helper._squeeze_helper(g, nonzero(g, index), [1])
    return index
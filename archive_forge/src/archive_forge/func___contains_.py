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
@_onnx_symbolic('aten::__contains_')
@_beartype.beartype
def __contains_(g: jit_utils.GraphContext, self, element):
    unpacked_list = symbolic_helper._unpack_list(self)
    if all((symbolic_helper._is_constant(x) for x in unpacked_list)) and symbolic_helper._is_constant(element):
        return g.op('Constant', value_t=torch.tensor(symbolic_helper._node_get(element.node(), 'value') in (symbolic_helper._node_get(x.node(), 'value') for x in unpacked_list)))
    raise errors.SymbolicValueError('Unsupported: ONNX export of __contains__ for non-constant list or element.', self)
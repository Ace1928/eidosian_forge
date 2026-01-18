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
@_onnx_symbolic('aten::_conj')
@_onnx_symbolic('aten::conj_physical')
@_beartype.beartype
def unsupported_complex_operators(g: jit_utils.GraphContext, input: _C.Value):
    if symbolic_helper.is_complex_value(input):
        return symbolic_helper._onnx_unsupported('aten::_conj, aten::conj_physical', input)
    return noop_complex_operators(g, input)
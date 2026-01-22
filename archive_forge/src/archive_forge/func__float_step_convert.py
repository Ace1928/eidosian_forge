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
def _float_step_convert(range_tensor):
    if symbolic_helper._is_fp(range_tensor):
        range_tensor = g.op('Cast', g.op('Ceil', range_tensor), to_i=_type_utils.JitScalarType.INT64.onnx_type())
    return range_tensor
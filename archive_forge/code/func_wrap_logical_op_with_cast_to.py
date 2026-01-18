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
def wrap_logical_op_with_cast_to(to_type):

    def decorator(fn):

        @functools.wraps(fn)
        def wrap_with_cast(g, input, other):
            to_cast_func = globals()[f'_cast_{to_type}']
            return fn(g, to_cast_func(g, input, False), to_cast_func(g, other, False))
        return wrap_with_cast
    return decorator
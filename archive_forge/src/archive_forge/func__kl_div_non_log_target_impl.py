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
def _kl_div_non_log_target_impl(g: jit_utils.GraphContext, input, target):
    log_ = log(g, target)
    diff_ = sub(g, log_, input)
    output_pos = mul(g, target, diff_)
    zeros_ = zeros_like(g, output_pos)
    mask_ = gt(g, target, g.op('Constant', value_t=torch.tensor(0)))
    output = where(g, mask_, output_pos, zeros_)
    return output
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
@_onnx_symbolic('aten::alpha_dropout_', decorate=[_apply_params('aten::alpha_dropout_')])
@_onnx_symbolic('aten::feature_alpha_dropout_', decorate=[_apply_params('aten::feature_alpha_dropout_')])
@_onnx_symbolic('aten::feature_dropout_', decorate=[_apply_params('aten::feature_dropout_')])
@_onnx_symbolic('aten::feature_alpha_dropout', decorate=[_apply_params('aten::feature_alpha_dropout')])
@_onnx_symbolic('aten::alpha_dropout', decorate=[_apply_params('aten::alpha_dropout')])
@_onnx_symbolic('aten::feature_dropout', decorate=[_apply_params('aten::feature_dropout')])
@_beartype.beartype
def _unsupported_dropout(name: str):

    @symbolic_helper.parse_args('v', 'none', 'b')
    @_beartype.beartype
    def feature_dropout(g, input, p, train):
        if train:
            return symbolic_helper._unimplemented(name, 'training mode', input)
        return input
    return feature_dropout
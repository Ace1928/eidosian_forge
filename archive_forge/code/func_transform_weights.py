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
def transform_weights(layer_index):
    weights = layer_weights[layer_index]
    if variant == 'RNN':
        weight_ih, weight_hh, bias_ih, bias_hh = weights
    elif variant == 'GRU' or variant == 'LSTM':
        weight_ih, weight_hh, bias_ih, bias_hh = (reform_weights(g, w, hidden_size, reform_permutation) for w in weights)
    bias_concat = g.op('Concat', bias_ih, bias_hh, axis_i=0)
    return tuple((symbolic_helper._unsqueeze_helper(g, x, [0]) for x in (weight_ih, weight_hh, bias_concat)))
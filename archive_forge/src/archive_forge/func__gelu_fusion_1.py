import functools
import operator
from functools import reduce
from typing import Any, Tuple
import torch
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from .. import ir
from ..lowering import lowerings as L
from ..pattern_matcher import (
from ..virtualized import ops
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
from .quantization import (
def _gelu_fusion_1(computation_call):
    return CallFunction(aten.mul, CallFunction(aten.mul, computation_call, 0.5), CallFunction(aten.add, CallFunction(aten.erf, CallFunction(aten.mul, computation_call, 0.7071067811865476)), 1))
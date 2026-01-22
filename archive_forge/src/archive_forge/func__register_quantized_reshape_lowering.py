import copy
import functools
import math
import operator
from typing import Any, Tuple
import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from ..lowering import lowerings as L, require_channels_last
from ..pattern_matcher import Arg, CallFunction, filter_nodes, KeywordArg, ListOf, Match
from ..utils import pad_listlike
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
def _register_quantized_reshape_lowering(pattern, computation_op):

    @register_lowering_pattern(pattern, extra_check=_is_input_output_same_scale_zp(aten.reshape.default))
    def qreshape(match: Match, *args, **kwargs):
        qx = kwargs['x']
        shape = kwargs['shape']
        counters['inductor']['qreshape_matcher_count'] += 1
        counters['inductor']['qreshape_matcher_nodes'] += len(match.nodes)
        return L[computation_op](qx, shape)
    return qreshape
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
def _is_packable_mkldnn_rnn_layer(match):
    lstm_node = match.output_node()
    POS_WEIGHTS = [1, 2]
    POS_INPUTS = [0, 5, 6]
    POS_ARGS = POS_WEIGHTS + POS_INPUTS
    if any((lstm_node.args[POS_WEIGHT].op != 'get_attr' for POS_WEIGHT in POS_WEIGHTS)):
        return False
    if any((lstm_node.args[POS_ARG].meta.get('val') is None for POS_ARG in POS_ARGS)):
        return False
    if any((lstm_node.args[POS_ARG].meta.get('val').device.type != 'cpu' for POS_ARG in POS_ARGS)):
        return False
    if any((lstm_node.args[POS_ARG].meta.get('val').dtype == torch.bfloat16 and (not mkldnn._is_mkldnn_bf16_supported()) for POS_ARG in POS_ARGS)):
        return False
    return True
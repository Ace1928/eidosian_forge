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
def _register_quantization_maxpool2d():
    max_pool2d_args_list = [[KeywordArg('stride')], [KeywordArg('stride'), KeywordArg('padding')], [KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation')], [KeywordArg('stride'), KeywordArg('padding'), KeywordArg('dilation'), KeywordArg('ceil_mode')]]
    for max_pool2d_args in max_pool2d_args_list:
        dequantize_maxpool2d_pattern = CallFunction(aten.max_pool2d_with_indices.default, dequantize_per_tensor_activation_pattern, KeywordArg('kernel_size'), *max_pool2d_args)
        dequantize_maxpool2d_get_item_pattern = CallFunction(operator.getitem, dequantize_maxpool2d_pattern, Arg())
        _register_quantized_maxpool2d_lowering(generate_pattern_with_output_quant(dequantize_maxpool2d_get_item_pattern), quantized.max_pool2d.default)
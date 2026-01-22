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
def _is_valid_dequant_promotion_pattern(dtype=torch.float32):

    def _inner(match):
        assert dtype in [torch.float32, torch.bfloat16]
        if dtype == torch.float32:
            mul_node = match.output_node()
        else:
            convert_to_bf16_node = match.output_node()
            mul_node = convert_to_bf16_node.args[0]
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        if mul_node.target is aten.mul.Tensor and sub_node.target is aten.sub.Tensor and (to_fp32_node.target is prims.convert_element_type.default) and (len(list(mul_node.users)) > 1) if dtype == torch.float32 else len(list(convert_to_bf16_node.users)) > 1:
            return True
        return False
    return _inner
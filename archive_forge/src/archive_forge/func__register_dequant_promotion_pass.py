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
def _register_dequant_promotion_pass(pattern, pass_number, dtype=torch.float32):

    @register_freezing_graph_pattern(pattern, extra_check=_is_valid_dequant_promotion_pattern(dtype), pass_number=pass_number)
    def dequant_promotion(match: Match, *args, **kwargs):
        assert dtype in [torch.float32, torch.bfloat16]

        def clone_to_new_node(graph, source_node, user_node):
            assert source_node.op == 'call_function', 'clone_to_new_node only support node.op call_function'
            with graph.inserting_before(user_node):
                new_node = graph.call_function(source_node.target, args=source_node.args, kwargs=source_node.kwargs)
                new_node.meta = copy.copy(source_node.meta)
                user_node.replace_input_with(source_node, new_node)
            return new_node
        if dtype == torch.float32:
            mul_node = match.output_node()
        else:
            convert_to_bf16_node = match.output_node()
            mul_node = convert_to_bf16_node.args[0]
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        assert mul_node.target is aten.mul.Tensor
        assert sub_node.target is aten.sub.Tensor
        assert to_fp32_node.target is prims.convert_element_type.default
        graph = match.graph
        user_node_list = list(mul_node.users) if dtype == torch.float32 else list(convert_to_bf16_node.users)
        for user_node in user_node_list:
            if dtype == torch.float32:
                new_mul_node = clone_to_new_node(graph, mul_node, user_node)
            else:
                new_convert_to_bf16_node_node = clone_to_new_node(graph, convert_to_bf16_node, user_node)
                new_mul_node = clone_to_new_node(graph, mul_node, new_convert_to_bf16_node_node)
            new_sub_node = clone_to_new_node(graph, sub_node, new_mul_node)
            _ = clone_to_new_node(graph, to_fp32_node, new_sub_node)
        counters['inductor']['dequant_promotion_matcher_count'] += 1
        counters['inductor']['dequant_promotion_matcher_nodes'] += len(match.nodes)
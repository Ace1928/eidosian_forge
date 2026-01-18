import functools
import torch
from torch._inductor.compile_fx import fake_tensor_prop
from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import (
@register_graph_pattern(CallFunction(torch.ops.prims.convert_element_type.default, Ignored(), KeywordArg('dtype')), pass_dict=pass_patterns[0], extra_check=same_dtype)
def unnecessary_dtype_convert(match: Match, **kwargs):
    """Remove unnecessary dtype conversion op, probably left as a result of Conv-Bn folding"""
    graph = match.graph
    node = match.output_node()
    node.replace_all_uses_with(node.args[0])
    graph.erase_node(node)
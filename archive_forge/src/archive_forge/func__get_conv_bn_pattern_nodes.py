import dataclasses
import itertools
import operator
from typing import Any, Callable, Dict, List, Tuple
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.subgraph_rewriter import (
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import (
from .utils import (
def _get_conv_bn_pattern_nodes(r: ReplacedPatterns) -> Dict[str, Tuple[Node, Node]]:
    """
    Helper function to extract the nodes in the conv-bn fusion pattern after
    subgraph rewriting, in the form of a map:

        {name: (original_node, replacement_node)}

    The following names must exist in the map:

        "conv", "conv_weight", "conv_input", "bn", "getitem"

    The following names may exist in the map:

        "conv_weight_q", "conv_weight_dq", "conv_bias",
        "conv_bias_q", "conv_bias_dq"
    """

    def _get_nodes(nodes: List[Node]) -> Tuple[Node, Node, Node]:
        """
        Return a 3-tuple of (conv_node, bn_node, getitem_node).
        This asserts that the match contains exactly one of each node.
        """
        conv_node, bn_node, getitem_node = (None, None, None)
        for n in nodes:
            if n.op != 'call_function':
                continue
            if _is_conv(n):
                assert conv_node is None
                conv_node = n
            if _is_supported_batch_norm_for_training(n) or n.target == torch.ops.aten._native_batch_norm_legit_no_training.default:
                assert bn_node is None
                bn_node = n
            if n.target == operator.getitem:
                assert getitem_node is None
                getitem_node = n
        assert conv_node is not None
        assert bn_node is not None
        assert getitem_node is not None
        return (conv_node, bn_node, getitem_node)

    def _get_q_dq_nodes(n: Node) -> Tuple[Node, Node, Node]:
        """
        Return a 3-tuple of (orig_node, q_node, dq_node).
        """
        assert _is_dequantize(n)
        q_node = n.args[0]
        assert isinstance(q_node, Node)
        assert _is_quantize(q_node)
        orig_node = q_node.args[0]
        assert isinstance(orig_node, Node)
        return (orig_node, q_node, n)
    original_nodes = list(_filter_nodes_map(r.nodes_map).values())
    o_conv, o_bn, o_getitem = _get_nodes(original_nodes)
    r_conv, r_bn, r_getitem = _get_nodes(r.replacements)
    mapping = {'conv': (o_conv, r_conv), 'bn': (o_bn, r_bn), 'getitem': (o_getitem, r_getitem)}
    p_conv, _, _ = _get_nodes(list(r.nodes_map.keys()))
    p_conv_input, p_conv_weight, *_ = p_conv.args
    r_conv_input, r_conv_weight, *_ = r_conv.args
    assert isinstance(p_conv_input, Node)
    assert isinstance(p_conv_weight, Node)
    assert isinstance(r_conv_input, Node)
    assert isinstance(r_conv_weight, Node)
    o_conv_input = r.nodes_map[p_conv_input]
    o_conv_weight = r.nodes_map[p_conv_weight]
    if _is_dequantize(p_conv_weight):
        p_conv_weight, p_conv_weight_q, p_conv_weight_dq = _get_q_dq_nodes(p_conv_weight)
        r_conv_weight, r_conv_weight_q, r_conv_weight_dq = _get_q_dq_nodes(r_conv_weight)
        o_conv_weight = r.nodes_map[p_conv_weight]
        o_conv_weight_q = r.nodes_map[p_conv_weight_q]
        o_conv_weight_dq = r.nodes_map[p_conv_weight_dq]
        mapping['conv_weight_q'] = (o_conv_weight_q, r_conv_weight_q)
        mapping['conv_weight_dq'] = (o_conv_weight_dq, r_conv_weight_dq)
    mapping['conv_input'] = (o_conv_input, r_conv_input)
    mapping['conv_weight'] = (o_conv_weight, r_conv_weight)
    if len(p_conv.args) > 2 and len(r_conv.args) > 2:
        p_conv_bias = p_conv.args[2]
        r_conv_bias = r_conv.args[2]
        assert isinstance(p_conv_bias, Node)
        assert isinstance(r_conv_bias, Node)
        o_conv_bias = r.nodes_map[p_conv_bias]
        if _is_dequantize(p_conv_bias):
            p_conv_bias, p_conv_bias_q, p_conv_bias_dq = _get_q_dq_nodes(p_conv_bias)
            r_conv_bias, r_conv_bias_q, r_conv_bias_dq = _get_q_dq_nodes(r_conv_bias)
            o_conv_bias = r.nodes_map[p_conv_bias]
            o_conv_bias_q = r.nodes_map[p_conv_bias_q]
            o_conv_bias_dq = r.nodes_map[p_conv_bias_dq]
            mapping['conv_bias_q'] = (o_conv_bias_q, r_conv_bias_q)
            mapping['conv_bias_dq'] = (o_conv_bias_dq, r_conv_bias_dq)
        mapping['conv_bias'] = (o_conv_bias, r_conv_bias)
    return mapping
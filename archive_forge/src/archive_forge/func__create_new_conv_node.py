import functools
import itertools
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern
def _create_new_conv_node(graph, conv_node, binary_node, other):
    assert conv_node.target == aten.convolution.default
    conv_args = list(conv_node.args)
    weight_meta_value = conv_node.args[1].meta.get('val')
    bias = conv_args[2]
    if binary_node.target in [aten.add.Tensor, aten.sub.Tensor]:
        other_reshape = resize_scalar_or_tensor_to_shape(graph, other, (weight_meta_value.size(0),))
        new_bias = graph.create_node('call_function', binary_node.target, (0 if bias is None else bias, other_reshape))
        conv_args[2] = new_bias
    else:
        assert binary_node.target in [aten.mul.Tensor, aten.div.Tensor]
        weight_broadcast_shape = [1 for _ in range(len(weight_meta_value.shape))]
        weight_broadcast_shape[0] = weight_meta_value.size(0)
        other_reshape1 = resize_scalar_or_tensor_to_shape(graph, other, tuple(weight_broadcast_shape))
        new_weight = graph.create_node('call_function', binary_node.target, (conv_args[1], other_reshape1))
        new_weight.meta.update(conv_args[1].meta)
        conv_args[1] = new_weight
        if bias is not None:
            other_reshape = resize_scalar_or_tensor_to_shape(graph, other, (weight_meta_value.size(0),))
            new_bias = graph.create_node('call_function', binary_node.target, (bias, other_reshape))
            new_bias.meta.update(bias.meta)
            conv_args[2] = new_bias
    return graph.create_node('call_function', conv_node.target, tuple(conv_args))
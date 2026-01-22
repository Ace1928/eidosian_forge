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
def _register_qconv_weight_prepack_pass(pattern, pass_number, dtype=torch.float32):

    @register_freezing_graph_pattern(pattern, extra_check=_is_valid_dequant_conv2d_pattern(dtype), pass_number=pass_number)
    def qconv_weight_prepack(match: Match, *args, **kwargs):
        """
        Match the pattern:
        int8 activation
          |
        dequant_per_tensor
          |
        Conv2d <- optional(aten.clone.default) <- dequant_per_channel <- int8_weight

        Insert weight prepack node and change the pattern to:
        int8 activation
          |
        onednn.qconv2d_pointwise <- onednn.qconv_prepack <- int8_weight
        """
        assert dtype in [torch.float32, torch.bfloat16]
        conv_node = match.output_node()
        assert conv_node.target is aten.convolution.default
        if dtype == torch.float32:
            mul_node = conv_node.args[0]
        else:
            convert_to_bf16 = conv_node.args[0]
            mul_node = convert_to_bf16.args[0]
        sub_node = mul_node.args[0]
        to_fp32_node = sub_node.args[0]
        has_clone_to_channel_last_node_in_pattern = conv_node.args[1].target is aten.clone.default
        clone_node = conv_node.args[1] if has_clone_to_channel_last_node_in_pattern else None
        if dtype == torch.float32:
            dequant_per_channel = clone_node.args[0] if has_clone_to_channel_last_node_in_pattern else conv_node.args[1]
        else:
            weight_to_bf16_node = clone_node.args[0] if has_clone_to_channel_last_node_in_pattern else conv_node.args[1]
            dequant_per_channel = weight_to_bf16_node.args[0]
        assert dequant_per_channel.target is quantized_decomposed.dequantize_per_channel.default
        qx, x_zp, x_scale = (kwargs['x'], kwargs['x_zp'], kwargs['x_scale'])
        qw, w_scale, w_zp = (kwargs['q_weight'], kwargs['w_scale'], kwargs['w_zp'])
        bias, stride, padding, dilation, groups = (kwargs['b'], kwargs['stride'], kwargs['padding'], kwargs['dilation'], kwargs['groups'])
        x_shape = qx.meta.get('tensor_meta').shape
        if has_free_symbols(x_shape):
            x_shape = None
        graph = match.graph
        with graph.inserting_before(conv_node):
            packed_weight_inputs = (qw, w_scale, x_scale, x_zp, stride, padding, dilation, groups, x_shape)
            packed_weight_op = torch.ops.onednn.qconv_prepack
            prepack_weight_node = graph.call_function(packed_weight_op, args=packed_weight_inputs)
            new_args: Tuple[Any, ...] = (qx, x_scale, x_zp, prepack_weight_node, w_scale, w_zp, bias, stride, padding, dilation, groups, 1.0, 0, dtype, 'none', [], '')
            new_conv_node = graph.call_function(torch.ops.onednn.qconv2d_pointwise.default, args=new_args)
            conv_node.replace_all_uses_with(new_conv_node)
            new_conv_node.meta.update(conv_node.meta)
            graph.erase_node(conv_node)
            if dtype == torch.bfloat16:
                graph.erase_node(convert_to_bf16)
            graph.erase_node(mul_node)
            graph.erase_node(sub_node)
            graph.erase_node(to_fp32_node)
            if clone_node is not None:
                graph.erase_node(clone_node)
            if dtype == torch.bfloat16:
                graph.erase_node(weight_to_bf16_node)
            graph.erase_node(dequant_per_channel)
            counters['inductor']['qconv2d_weight_prepack_matcher_count'] += 1
            counters['inductor']['qconv2d_weight_prepack_matcher_nodes'] += len(match.nodes)
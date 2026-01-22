import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
@register_fusion('batch_layernorm')
class BatchLayernormFusion(BatchFusion):
    """
    Batch layer norm fusion in pre grad pass
    """

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(torch.nn.functional.layer_norm).match(node):
            input = get_arg_value(node, 0, 'input')
            weight = get_arg_value(node, 2, 'weight')
            bias = get_arg_value(node, 3, 'bias')
            group_key = ('batch_layernorm', str(input.meta['example_value'].shape), str(weight.meta['example_value'].shape) if weight is not None else '', str(bias.meta['example_value'].shape) if bias is not None else '', str(get_arg_value(node, 1, 'normalized_shape')), str(get_arg_value(node, 4, 'eps'))) if 'example_value' in input.meta and is_node_meta_valid(weight) and is_node_meta_valid(bias) else None
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        group_inputs = []
        group_shapes = []
        group_weights = []
        group_biases = []
        group_epss = []
        group_nodes = []
        for node in subset:
            group_nodes.append(node)
            group_inputs.append(get_arg_value(node, 0, 'input'))
            group_shapes.append(get_arg_value(node, 1, 'normalized_shape'))
            group_weights.append(get_arg_value(node, 2, 'weight'))
            group_biases.append(get_arg_value(node, 3, 'bias'))
            eps = get_arg_value(node, 4, 'eps')
            if eps is None:
                eps = 1e-05
            group_epss.append(eps)
        stack_dim = -1 - len(group_shapes[-1])
        if all((bias is None for bias in group_biases)):
            group_biases = None
        group_biases: Optional[List[Any]]
        if all((weight is None for weight in group_weights)):
            group_weights = None
        group_weights: Optional[List[Any]]
        assert all((eps == group_epss[0] for eps in group_epss)), 'all epsilon values must be equal'
        with graph.inserting_before(subset[0]):
            stack_input = graph.call_function(torch.stack, args=(group_inputs,), kwargs={'dim': stack_dim})
            if group_weights is not None:
                stack_weight = graph.call_function(torch.stack, args=(group_weights,), kwargs={'dim': 0})
            else:
                stack_weight = None
            if group_biases is not None:
                stack_bias = graph.call_function(torch.stack, args=(group_biases,), kwargs={'dim': 0})
            else:
                stack_bias = None
            batch_layer_norm = graph.call_function(torch.nn.functional.layer_norm, args=(stack_input, group_shapes[-1]), kwargs={'eps': group_epss[-1]})
            if group_weights is not None and group_biases is not None:
                batch_layer_norm = graph.call_function(torch.addcmul, args=(stack_bias, stack_weight, batch_layer_norm))
            elif group_weights is not None and group_biases is None:
                batch_layer_norm = graph.call_function(torch.mul, args=(stack_weight, batch_layer_norm))
            elif group_weights is None and group_biases is not None:
                batch_layer_norm = graph.call_function(torch.add, args=(stack_bias, batch_layer_norm))
            batch_layer_norm_unbind = graph.call_function(torch.unbind, args=(batch_layer_norm,), kwargs={'dim': stack_dim})
        for i, node in enumerate(group_nodes):
            with graph.inserting_after(batch_layer_norm_unbind):
                new_node = graph.call_function(operator.getitem, args=(batch_layer_norm_unbind, i))
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)
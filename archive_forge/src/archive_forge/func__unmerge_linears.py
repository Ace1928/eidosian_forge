import collections
import copy
import functools
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
import torch
from transformers.file_utils import add_end_docstrings
from transformers.utils.fx import _gen_constructor_wrapper
@staticmethod
def _unmerge_linears(graph_module: 'GraphModule', merged_linear_node: 'Node', merged_linear: torch.nn.Linear):
    linear_node_targets = merged_linear_node.linear_node_targets
    output_nodes = sorted(merged_linear_node.users, key=lambda node: node.args[1][1].start)
    in_features = merged_linear.in_features
    out_features = []
    for node in output_nodes:
        slice_to_get = node.args[1][1]
        out_features.append(slice_to_get.stop - slice_to_get.start)
    linears = [torch.nn.Linear(in_features, out_feat, bias=hasattr(merged_linear, 'bias'), device=merged_linear.weight.device, dtype=merged_linear.weight.dtype) for out_feat in out_features]
    for target, node, linear in zip(linear_node_targets, output_nodes, linears):
        with torch.no_grad():
            slice_to_get = node.args[1][1]
            linear.weight = torch.nn.Parameter(merged_linear.weight[slice_to_get.start:slice_to_get.stop])
            if hasattr(merged_linear, 'bias'):
                linear.bias = torch.nn.Parameter(merged_linear.bias[slice_to_get.start:slice_to_get.stop])
        parent_module, name = MergeLinears._linear_node_to_module_and_attribute_name(graph_module, target)
        parent_module.add_module(name, linear)
        node.op = 'call_module'
        node.target = target
        node.args = (merged_linear_node.args[0],)
    parent_module, merged_linear_name = MergeLinears._linear_node_to_module_and_attribute_name(graph_module, merged_linear_node.target)
    delattr(parent_module, merged_linear_name)
    graph_module.graph.erase_node(merged_linear_node)
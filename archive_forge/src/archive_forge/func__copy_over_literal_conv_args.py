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
def _copy_over_literal_conv_args(original_node: Node, new_node: Node):
    """
    Copy over literal args in conv, such as stride and padding, from the matched node
    in the original graph to its replacement in the new graph.

    This is needed due to the following limitation in the subgraph rewriter when used
    with dynamo export: literal (non-tensor) args are not supported in the match and
    replacement patterns. This is because dynamo export automatically inlines these
    literal args, making them dead placeholder nodes. In the future, we should check
    if dynamo export can optionally disable this inlining, or if subgraph rewriter
    can do the copying for us. See https://github.com/pytorch/pytorch/issues/100419.

    Note: Unlike other tensor args like conv weights and biases, literal args are
    preserved in the original nodes after replacement, so we can access them here.
    """
    assert _is_conv(original_node)
    assert _is_conv(new_node)
    new_args = list(new_node.args)
    if len(new_args) < 3:
        new_args.append(None)
    new_node.args = tuple(new_args[:3]) + original_node.args[3:]
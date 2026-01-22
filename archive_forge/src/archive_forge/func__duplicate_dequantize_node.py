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
def _duplicate_dequantize_node(m: GraphModule):
    """
    Helper function to duplicate all dequantize nodes in the graph if the
    node has more than one user. For example:

    Before:
      quantize -> dequantize -> a
                          \\--> b
                          \\--> c

    After:
      quantize -> dequantize_1 -> a
            \\--> dequantize_2 -> b
            \\--> dequantize_3 -> c

    This is useful for subgraph rewriting. E.g. if we wish to match the
    pattern [dequantize - a] above, subgraph matching would fail because
    the dequantize node has users outside the matched portion of the graph.
    Instead, we match [dequantize_1 - a], which is safe.
    """
    dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor
    for n in m.graph.nodes:
        if n.op != 'call_function' or n.target != dq_op or len(n.users) == 1:
            continue
        for user in list(n.users):
            with m.graph.inserting_before(n):
                new_node = m.graph.create_node('call_function', dq_op, n.args, n.kwargs)
            user.replace_input_with(n, new_node)
        m.graph.erase_node(n)
    m.recompile()
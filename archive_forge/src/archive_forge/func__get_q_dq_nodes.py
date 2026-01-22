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
import contextlib
from typing import List, Tuple
import torch
def _get_differentiable_graph_node(node, diff_node):
    if node.kind() == 'prim::DifferentiableGraph':
        diff_node.append(node)
    else:
        for block in node.blocks():
            for n in block.nodes():
                _get_differentiable_graph_node(n, diff_node)
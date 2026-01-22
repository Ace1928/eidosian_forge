import torch
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from .utils import (
from .ns_types import (
from torch.ao.ns.fx.mappings import (
from torch.ao.quantization.observer import _is_activation_post_process
from typing import Dict, Tuple, Callable, List, Any, Union, Optional, Set
def _insert_copy_of_subgraph_a_after_input_node_c(input_node_c: Union[Node, List[Node]], input_node_c_2: Optional[Union[Node, List[Node]]], subgraph_a: NSSubgraph, gm_a: GraphModule, gm_b: GraphModule, node_name_prefix: str) -> Node:
    """
    TODO(before land): real docblock
    """
    if isinstance(input_node_c, Node):
        graph_c = input_node_c.graph
    else:
        assert isinstance(input_node_c, list)
        graph_c = input_node_c[0].graph
    nodes_of_a = [subgraph_a.end_node]
    cur_node = subgraph_a.end_node
    while cur_node != subgraph_a.start_node:
        cur_node = get_normalized_nth_input(cur_node, gm_a, 0)
        nodes_of_a.insert(0, cur_node)
    cur_node_a = nodes_of_a[0]
    cur_node_c = _insert_copy_of_node_a_after_input_node_c(input_node_c, input_node_c_2, cur_node_a, gm_a, gm_b, node_name_prefix)
    for cur_idx_a in range(1, len(nodes_of_a)):
        cur_node_a = nodes_of_a[cur_idx_a]
        prev_node_c = cur_node_c
        cur_node_c = _insert_copy_of_node_a_after_input_node_c(prev_node_c, None, cur_node_a, gm_a, gm_b, node_name_prefix)
    return cur_node_c
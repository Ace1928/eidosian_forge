import copy
import inspect
import logging
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Type
import torch.nn as nn
from torch import fx
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.partial_lower import partial_lower
from torch.fx.graph import _PyTreeCodeGen, PythonCode
from torch.fx.node import Argument
from torch.profiler import record_function
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
def _forward_inputs_to_subgraph(self, subgraph: List[fx.Node], graph: fx.Graph, extra_input: int) -> None:
    """Create extra input nodes and forward the input nodes to the ``subgraph``.

        The external input nodes of ``subgraph`` (nodes that are not in ``subgraph``) will replaced by the newly
        created input nodes.
        """
    placeholders = [node for node in graph.nodes if str(node.op) == 'placeholder']
    assert placeholders, 'No placeholders are found'
    with self._fx_graph_call(graph, 'inserting_after', placeholders[-1]):
        new_input_nodes = list(reversed([self._fx_graph_call(graph, 'placeholder', f'cross_iter_input_{self._cross_iter_block_count}_{i}') for i in reversed(range(extra_input))]))
    all_nodes = set(subgraph)
    new_input_index = 0
    for node in subgraph:
        node_inputs, spec = tree_flatten((node.args, node.kwargs))
        new_node_inputs = []
        for input_node in node_inputs:
            if not isinstance(input_node, fx.Node) or input_node in all_nodes:
                new_node_inputs.append(input_node)
            else:
                new_node_inputs.append(new_input_nodes[new_input_index])
                new_input_index += 1
        node.args, node.kwargs = tree_unflatten(new_node_inputs, spec)
    assert new_input_index == len(new_input_nodes), f'More inputs than needed {len(new_input_nodes)} > {new_input_index}'
    if isinstance(graph._codegen, _PyTreeCodeGen) and graph._codegen.pytree_info.in_spec is not None:
        codegen = graph._codegen
        original_tree_in = tree_unflatten(placeholders, codegen.pytree_info.in_spec)
        _, in_spec = tree_flatten(tuple(list(original_tree_in) + new_input_nodes))
        codegen.pytree_info = codegen.pytree_info._replace(in_spec=in_spec)
        for new_input in new_input_nodes:
            codegen.pytree_info.orig_args.append(new_input.name)
        codegen.pytree_info = codegen.pytree_info._replace(in_spec=in_spec)
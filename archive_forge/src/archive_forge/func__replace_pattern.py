from .graph_module import GraphModule
from .graph import Graph
from .node import Node
from ._symbolic_trace import symbolic_trace
from ._compatibility import compatibility
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union
import torch
def _replace_pattern(gm: GraphModule, pattern: Union[Callable, Graph, GraphModule], replacement: Union[Callable, Graph, GraphModule], match_filters: Optional[List[Callable[['InternalMatch', Graph, Graph], bool]]]=None, ignore_literals: bool=False) -> List[ReplacedPatterns]:
    from torch.fx.passes.utils.matcher_utils import SubgraphMatcher, InternalMatch
    if match_filters is None:
        match_filters = []
    original_graph: Graph = gm.graph
    if isinstance(pattern, GraphModule):
        pattern_graph = pattern.graph
    elif isinstance(pattern, Graph):
        pattern_graph = pattern
    else:
        pattern_graph = symbolic_trace(pattern).graph
    if isinstance(replacement, GraphModule):
        replacement_graph = replacement.graph
    elif isinstance(replacement, Graph):
        replacement_graph = replacement
    else:
        replacement_graph = symbolic_trace(replacement).graph
    matcher = SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False, remove_overlapping_matches=True, ignore_literals=ignore_literals)
    _matches: List[InternalMatch] = matcher.match(original_graph)
    _matches = [m for m in _matches if all((match_filter(m, original_graph, pattern_graph) for match_filter in match_filters))]
    replacement_placeholders = [n for n in replacement_graph.nodes if n.op == 'placeholder']
    match_changed_node: Dict[Node, Node] = {}
    match_and_replacements = []
    for match in _matches:
        assert len(match.placeholder_nodes) == len(replacement_placeholders)
        val_map: Dict[Node, Node] = {}
        for rn, gn in zip(replacement_placeholders, match.placeholder_nodes):
            if isinstance(gn, Node):
                val_map[rn] = match_changed_node.get(gn, gn)
                if gn != val_map[rn]:
                    gn_ind = match.placeholder_nodes.index(gn)
                    match.placeholder_nodes[gn_ind] = match_changed_node[gn]
                    map_key = list(match.nodes_map.keys())[list(match.nodes_map.values()).index(gn)]
                    match.nodes_map[map_key] = match_changed_node[gn]
            else:
                val_map[rn] = gn
        user_nodes: Set[Node] = set()
        for n in match.returning_nodes:
            for user in n.users:
                user_nodes.add(user)
        assert user_nodes, 'The returning_nodes should have at least one user node'
        if len(user_nodes) == 1:
            first_user_node = next(iter(user_nodes))
        else:
            for n in original_graph.nodes:
                if n in user_nodes:
                    first_user_node = n
                    break
        with original_graph.inserting_before(first_user_node):
            copied_returning_nodes = original_graph.graph_copy(replacement_graph, val_map)
        if isinstance(copied_returning_nodes, Node):
            copied_returning_nodes = (copied_returning_nodes,)
        replacement_nodes: List[Node] = [v for v in val_map.values() if v not in match.placeholder_nodes]
        assert len(match.returning_nodes) == len(copied_returning_nodes)
        for gn, copied_node in zip(match.returning_nodes, copied_returning_nodes):
            gn.replace_all_uses_with(copied_node)
            match_changed_node[gn] = copied_node
        for node in reversed(pattern_graph.nodes):
            if node.op != 'placeholder' and node.op != 'output':
                gn = match.nodes_map[node]
                gm.graph.erase_node(gn)
        match_and_replacements.append(ReplacedPatterns(anchor=match.anchors[0], nodes_map=match.nodes_map, replacements=replacement_nodes))
    gm.recompile()
    if isinstance(replacement, torch.nn.Module):
        _replace_attributes(gm, replacement)
    return match_and_replacements
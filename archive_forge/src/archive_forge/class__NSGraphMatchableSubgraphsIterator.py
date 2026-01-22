import collections
import enum
import torch
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node
from torch.ao.quantization.utils import getattr_from_fqn
from .ns_types import NSSubgraph, NSNodeTargetType
from .mappings import (
from .pattern_utils import (
from torch.ao.quantization import (
from typing import Dict, Tuple, List, Optional, Set, Any
class _NSGraphMatchableSubgraphsIterator:
    """
    Iterates through the graph of gm, starting with the output nodes
    and continuing backwards.
    1. Returns matchable subgraphs, in order. A subgraph is defined by
       (start_node, end_node).
    2. Skips over non-matchable subgraphs
    """

    def __init__(self, gm: GraphModule, non_matchable_functions: Set[NSNodeTargetType], non_matchable_modules: Set[NSNodeTargetType], non_matchable_methods: Set[NSNodeTargetType]):
        self.gm: GraphModule = gm
        self.non_matchable_functions: Set[NSNodeTargetType] = non_matchable_functions
        self.non_matchable_modules: Set[NSNodeTargetType] = non_matchable_modules
        self.non_matchable_methods: Set[NSNodeTargetType] = non_matchable_methods
        self.seen_nodes: Set[Node] = set()
        self.stack: List[Node] = []
        for start_node in _get_output_nodes(self.gm.graph):
            self.stack.append(start_node)

    def __iter__(self):
        return self

    def __next__(self) -> NSSubgraph:
        """
        Returns the next matchable subgraph.
        """
        while len(self.stack) > 0:
            cur_end_node = self.stack.pop()
            if cur_end_node in self.seen_nodes:
                continue
            cur_start_node = cur_end_node
            cur_base_op_node = cur_end_node
            for _reverse_fusion_ops, base_op_idx in get_reversed_fusions():
                is_match = end_node_matches_reversed_fusion(cur_end_node, _reverse_fusion_ops, self.gm, self.seen_nodes)
                if is_match:
                    for rev_fusion_idx in range(len(_reverse_fusion_ops) - 1):
                        self.seen_nodes.add(cur_start_node)
                        cur_start_node = cur_start_node.args[0]
                        rev_base_op_idx = len(_reverse_fusion_ops) - 2 - base_op_idx
                        if rev_fusion_idx == rev_base_op_idx:
                            cur_base_op_node = cur_start_node
                    break
            self.seen_nodes.add(cur_start_node)
            for arg in cur_start_node.all_input_nodes:
                self._recursively_add_node_arg_to_stack(arg)
            if not self._is_matchable(cur_base_op_node):
                continue
            if cur_end_node.op == 'call_module' and cur_start_node is cur_end_node:
                maybe_obs = getattr_from_fqn(self.gm, cur_end_node.target)
                if isinstance(maybe_obs, (ObserverBase, FakeQuantizeBase)):
                    continue
            return NSSubgraph(start_node=cur_start_node, end_node=cur_end_node, base_op_node=cur_base_op_node)
        raise StopIteration

    def _recursively_add_node_arg_to_stack(self, arg: Any) -> None:
        """
        Adds all of the nodes in this arg to the stack, properly navigating
        through list, dicts and tuples.
        """
        if isinstance(arg, Node):
            self.stack.append(arg)
        elif isinstance(arg, torch.fx.immutable_collections.immutable_list) or type(arg) is tuple:
            for inner_arg in arg:
                self._recursively_add_node_arg_to_stack(inner_arg)
        elif isinstance(arg, torch.fx.immutable_collections.immutable_dict):
            for value in arg.values():
                self._recursively_add_node_arg_to_stack(value)

    def _is_matchable(self, node: Node) -> bool:
        if node.op == 'call_function':
            return node.target not in self.non_matchable_functions
        elif node.op == 'call_module':
            assert isinstance(node.target, str)
            target_mod = getattr_from_fqn(self.gm, node.target)
            return not any((isinstance(target_mod, t) for t in self.non_matchable_modules))
        elif node.op == 'call_method':
            return node.target not in self.non_matchable_methods
        else:
            return False
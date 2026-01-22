from typing import List, Tuple, Union, Dict, Any, Set, Mapping
import collections
from dataclasses import dataclass
import torch
import torch.fx
from torch.fx.node import _get_qualified_name
from torch.fx._compatibility import compatibility
@compatibility(is_backward_compatible=False)
class FxNetAccFusionsFinder:
    """
    Finds groups of connected ACC nodes that pass non-tensor data between each other.
    Such groups are called fusion groups.
    """

    def __init__(self, module: torch.fx.GraphModule, acc_nodes: NodeSet):
        self.module = module
        self.nodes = list(module.graph.nodes)
        self.acc_nodes = acc_nodes

    @dataclass
    class FusionGroup:
        top_node_idx: int
        nodes: NodeSet
        inputs: NodeSet
        nodes_need_process: NodeSet

        def add_node(self, node):
            """
            Add a node to fusion group.
            """
            if node in self.nodes:
                return
            self.nodes_need_process.add(node)
            self.nodes.add(node)
            self.inputs.discard(node)
            self.inputs.update({n for n in node.all_input_nodes if n.op in CALLABLE_NODE_OPS and n not in self.nodes})

    def recursive_add_node(self, fusion_group: 'FxNetAccFusionsFinder.FusionGroup', inputs: Union[NodeSet, NodeList]):
        """
        Start from inputs and going reverse topological order. If any upstream node
        is in the fusion group, add all the nodes in this path to fusion group.
        """
        for arg in inputs:
            if arg.op not in CALLABLE_NODE_OPS:
                continue
            if self.nodes.index(arg) < fusion_group.top_node_idx:
                continue
            if arg in fusion_group.nodes:
                return True
            if self.recursive_add_node(fusion_group, arg.all_input_nodes):
                fusion_group.add_node(arg)
                return True
        return False

    def __call__(self) -> Dict[torch.fx.Node, NodeSet]:
        result: Dict[torch.fx.Node, NodeSet] = {}
        acc_nodes = list(self.acc_nodes)
        for node in acc_nodes:
            if node in result:
                continue
            if node.op not in CALLABLE_NODE_OPS:
                continue
            if 'tensor_meta' in node.meta:
                continue
            if node not in self.acc_nodes:
                continue
            fusion_group: FxNetAccFusionsFinder.FusionGroup = self.FusionGroup(top_node_idx=self.nodes.index(node), nodes={node}, inputs=set(node.all_input_nodes), nodes_need_process={node})
            while fusion_group.nodes_need_process:
                node = fusion_group.nodes_need_process.pop()
                self.recursive_add_node(fusion_group, fusion_group.inputs)
                if 'tensor_meta' not in node.meta:
                    for user in node.users:
                        if user.op not in CALLABLE_NODE_OPS:
                            continue
                        if user in fusion_group.nodes:
                            continue
                        fusion_group.add_node(user)
                        self.recursive_add_node(fusion_group, fusion_group.inputs)
                for arg in node.all_input_nodes:
                    if arg.op not in CALLABLE_NODE_OPS:
                        continue
                    if 'tensor_meta' in arg.meta:
                        continue
                    if arg in fusion_group.nodes:
                        continue
                    fusion_group.add_node(arg)
                    fusion_group.top_node_idx = min(fusion_group.top_node_idx, self.nodes.index(arg))
                    self.recursive_add_node(fusion_group, fusion_group.inputs)
            if not set(fusion_group.nodes) <= self.acc_nodes:
                self.acc_nodes -= fusion_group.nodes
            else:
                for n in fusion_group.nodes:
                    result[n] = fusion_group.nodes
        return result
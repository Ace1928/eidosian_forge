import torch
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from .utils import (
from .ns_types import (
from torch.ao.ns.fx.mappings import (
from torch.ao.quantization.observer import _is_activation_post_process
from typing import Dict, Tuple, Callable, List, Any, Union, Optional, Set
def _insert_logger_after_node(node: Node, gm: GraphModule, logger_cls: Callable, logger_node_name_suffix: str, ref_node_name: str, model_name: str, ref_name: str, ref_node_target_type: str, results_type: str, index_within_arg: int, index_of_arg: int, fqn: Optional[str]) -> Node:
    """
    Given a starting graph of

    prev_node -> node -> next_node

    This function creates a new logger_cls obj and adds it
    after node, resulting in

    prev_node -> node -> logger_obj -> next_node
    """
    logger_node_name = get_new_attr_name_with_prefix(node.name + logger_node_name_suffix)(gm)
    target_type = get_target_type_str(node, gm)
    logger_obj = logger_cls(ref_node_name, node.name, model_name, ref_name, target_type, ref_node_target_type, results_type, index_within_arg, index_of_arg, fqn)
    setattr(gm, logger_node_name, logger_obj)
    logger_node = node.graph.create_node('call_module', logger_node_name, (node,), {})
    return logger_node
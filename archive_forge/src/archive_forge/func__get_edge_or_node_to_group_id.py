import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.prepare import (
from torch.fx import (
from torch.fx.node import Argument
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Union, Any, Optional
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
def _get_edge_or_node_to_group_id(edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase]) -> Dict[EdgeOrNode, int]:
    """Map from edge/node to the group ID, generated from quantization annotations,
    edge/node with the same group ID should use the same observer/fake_quant instance

    This is applying SharedQuantizationSpec configuration and map each edge/node to a group
    There is another implicit sharing that's built in the quantization, when we have the following:
       * op1 -> op2
       * output of op1: int8_qspec
       * (op1 -> op2) input edge: int8_qspec
    we'll assume sharing between the output of op1 and input of (op1 -> op2) since these are the same Tensor.

    Figuring out the correct group ID for all edge/node is a standard union find problem:
    https://www.geeksforgeeks.org/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/

    Args:
        edge_or_node_to_qspec: Dictionary from edge_or_node to the qspec, derived from annotations
    Returns:
        edge_or_node_to_group_id: Dictionary from edge_or_node to group_id (int), all edge or node that
        belongs to the same group should have the same id

    Example:
        op2 -> cat1 -> cat2
           op1 /        /
                     op3
        edge_or_node_to_qspec: {
            op1: int8_qspec,
            op2: int8_qspec,
            (op1, cat1): int8_qspc,
            (op2, cat1): SharedQuantizationSpec((op1, cat1)),
            cat1: SharedQuantizationSpec((op1, cat1)),
            (op3, cat2): int8_qspec,
            (cat1, cat2): SharedQuantizationSpec((op3, cat2)),
            cat2: SharedQuantizationSpec((op3, cat2)),
        }

        edge_or_node_to_group_id = _get_edge_or_node_to_group_id(edge_or_node_to_qspec)
        edge_or_node_to_group_id: {
            op1: 1,
            op2: 1,
            (op1, cat1): 1,
            (op2, cat1): 1,
            cat1: 1,
            (op3, cat2): 1,
            (cat1, cat2): 1,
            cat2: 1,
        }
        # everything are in the same group because (cat1) and (cat1, cat2) are implicitly shared, which
        # connects the two sharing group around cat1 and cat2 op due to transitive sharing
    """
    shared_with_map: Dict[EdgeOrNode, EdgeOrNode] = {k: k for k in edge_or_node_to_qspec.keys()}
    for edge_or_node, qspec in edge_or_node_to_qspec.items():
        if isinstance(edge_or_node, torch.fx.Node):
            output_node = edge_or_node
            _update_shared_with(output_node, qspec, shared_with_map)
        else:
            input_edge = edge_or_node
            input_edge_root_qspec = _unwrap_shared_qspec(qspec, edge_or_node_to_qspec, shared_with_map)
            assert isinstance(input_edge, tuple)
            arg, n = input_edge
            if n.meta['quantization_annotation'].allow_implicit_sharing:
                for user in arg.users:
                    if user is n:
                        continue
                    arg_to_user_edge = (arg, user)
                    _union_input_edge_with(input_edge, input_edge_root_qspec, arg_to_user_edge, edge_or_node_to_qspec, shared_with_map)
                _union_input_edge_with(input_edge, input_edge_root_qspec, arg, edge_or_node_to_qspec, shared_with_map)
            _update_shared_with(input_edge, qspec, shared_with_map)
    cur_group_id = 0
    edge_or_node_to_group_id: Dict[EdgeOrNode, int] = {}
    for edge_or_node in shared_with_map.keys():
        root = _find_root_edge_or_node(edge_or_node, shared_with_map)
        if root not in edge_or_node_to_group_id:
            edge_or_node_to_group_id[root] = cur_group_id
            cur_group_id += 1
        edge_or_node_to_group_id[edge_or_node] = edge_or_node_to_group_id[root]
    return edge_or_node_to_group_id
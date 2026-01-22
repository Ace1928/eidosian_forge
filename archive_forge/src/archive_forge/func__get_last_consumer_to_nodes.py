import logging
import operator
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.distributed._spmd.experimental_ops
import torch.fx as fx
from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.distributed._spmd.graph_utils import OP
from torch.distributed._spmd.log_utils import get_logger
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.op_schema import OpSchema
from torch.distributed._tensor.placement_types import (
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
def _get_last_consumer_to_nodes(graph: fx.Graph) -> Dict[fx.Node, List[fx.Node]]:
    node_to_last_consumer: Dict[fx.Node, fx.Node] = {}
    last_consumer_to_nodes: Dict[fx.Node, List[fx.Node]] = {}

    def _register_final_consumer(arg_node: fx.Node, consumer: fx.Node) -> None:
        if arg_node not in node_to_last_consumer:
            node_to_last_consumer[arg_node] = consumer
            last_consumer_to_nodes.setdefault(consumer, []).append(arg_node)
    for node in reversed(graph.nodes):
        fx.node.map_arg(node.args, lambda arg_node: _register_final_consumer(arg_node, node))
        fx.node.map_arg(node.kwargs, lambda kwarg_node: _register_final_consumer(kwarg_node, node))
    return last_consumer_to_nodes
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
def _convert_to_distributed(gm: fx.GraphModule, inps: List[torch.Tensor], schemas: List[Schema], default_mesh: Optional[DeviceMesh]=None, _allow_partial: bool=False) -> Tuple[fx.GraphModule, Dict[str, Schema]]:
    """Transform a graph module to a distributed graph module.

    Returns:
        - transformed graph module
        - map from output name to DTensorSpec

    """
    global logger
    logger = get_logger('spmd_exp')
    operators = {getattr(operator, name) for name in operator.__all__}
    node_to_obj: Dict[fx.Node, Any] = {}
    node_replacements: Dict[torch.fx.Node, torch.fx.GraphModule] = {}
    last_consumer_to_nodes = _get_last_consumer_to_nodes(gm.graph)
    output_schemas: Dict[str, Schema] = {}
    for i, node in enumerate(gm.graph.nodes):
        assert logger is not None
        logger.info('node%s: op=%s target=%s', i, node.op, node.target)
        if node.op == OP.PLACEHOLDER:
            assert i < len(inps), f'got more placeholder nodes ({i + 1}) than inputs ({len(inps)})'
            node_to_obj[node] = DTensor.from_local(inps[i].clone(), schemas[i].mesh, schemas[i].placements, run_check=False)
        elif isinstance(node.target, torch._ops.OpOverloadPacket):
            dtensor = cast(DTensor, node_to_obj[node.args[0]])
            node_to_obj[node] = DSymInt.from_node(node, dtensor)
        elif isinstance(node.target, torch._ops.OpOverload):
            replacement = _get_dtensor_dispatch_graph(node, node_to_obj, default_mesh=default_mesh)
            if replacement is not None:
                node_replacements[node] = replacement
        elif node.op == OP.OUTPUT:
            if not _allow_partial:
                node = _convert_output(gm, node, node_to_obj)
            for inp_arg in node.args[0]:
                if isinstance(inp_arg, fx.Node):
                    obj = node_to_obj[inp_arg]
                    if isinstance(obj, DTensor):
                        output_schemas[inp_arg.name] = Schema(obj.device_mesh, obj.placements)
        elif node.op == OP.CALL_FUNCTION:
            args = tree_map(partial(_remap_arg, node_to_obj), node.args)
            kwargs = tree_map(partial(_remap_arg, node_to_obj), node.kwargs)
            dsymints = list(filter(lambda a: isinstance(a, DSymInt), args + tuple(kwargs.values())))
            if node.target in operators and len(dsymints) > 0:
                assert all((dsymints[0].mesh == d.mesh for d in dsymints)), 'all DSymInts must have the same mesh. '
                local_args = tree_map_only(DSymInt, lambda a: a.local_value, args)
                local_kwargs = tree_map_only(DSymInt, lambda a: a.local_value, kwargs)
                global_args = tree_map_only(DSymInt, lambda a: a.global_value, args)
                global_kwargs = tree_map_only(DSymInt, lambda a: a.global_value, kwargs)
                node.args = local_args
                node.kwargs = local_kwargs
                node_to_obj[node] = DSymInt(local_value=node.target(*local_args, **local_kwargs), global_value=node.target(*global_args, **global_kwargs), mesh=dsymints[0].mesh)
            else:
                assert len(dsymints) == 0, f'SPMD expansion does not support SymInt in non-operator nodes, got {node.target}.'
                node_to_obj[node] = node.target(*args, **kwargs)
        else:
            raise ValueError(f'Unrecognized node.op type {node.op}')
        if node in last_consumer_to_nodes:
            for arg_node in last_consumer_to_nodes[node]:
                del node_to_obj[arg_node]
    _rebuild_graph(gm, node_replacements)
    return (gm, output_schemas)
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
def _build_dummy_add_graph(dt: DTensor, node_to_obj: Dict[fx.Node, Any]) -> Tuple[fx.GraphModule, Any]:
    """Create a graph for a dummy add function from a partial DTensor.

    This dummy add is used for triggering all_reduce on a Partial DTensor
    during the DTensor expansion of the traced graph.
    Also returns the actual DTensor after resharding.
    """

    def dummy_add(grad: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        return grad + zero
    grad: torch.Tensor = dt._local_tensor
    zero: torch.Tensor = torch.zeros_like(dt._local_tensor)
    traced_add = make_fx(dummy_add)(grad, zero)
    placeholders = [n for n in traced_add.graph.nodes if n.op == OP.PLACEHOLDER]
    call_functions = [n for n in traced_add.graph.nodes if n.op == OP.CALL_FUNCTION]
    assert len(placeholders) == 2
    assert len(call_functions) == 1
    node_to_obj[placeholders[0]] = dt
    node_to_obj[placeholders[1]] = DTensor.from_local(zero, dt.device_mesh, [Replicate()], run_check=False)
    traced_dispatch = _get_dtensor_dispatch_graph(call_functions[0], node_to_obj, force_make_fx=True)
    assert traced_dispatch is not None
    return (traced_dispatch, node_to_obj[call_functions[0]])
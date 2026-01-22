import copy
import operator
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import (
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.distributed.tensor.parallel.style import ColwiseParallel, ParallelStyle
from torch.export import ExportedProgram
from torch.export.exported_program import ExportGraphSignature
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree
def _shard_state_dict(state_dict: Dict[str, torch.Tensor], placement_strategies: Dict[Node, PlacementStrategy], graph_signature: ExportGraphSignature, mesh: DeviceMesh) -> None:
    """
    Inplace partition the weights based on the placement strategy
    """
    for node, placement_strategy in placement_strategies.items():
        if node.op != 'placeholder':
            continue
        if node.name in graph_signature.inputs_to_parameters:
            fqn = graph_signature.inputs_to_parameters[node.name]
        elif node.name in graph_signature.inputs_to_buffers:
            fqn = graph_signature.inputs_to_buffers[node.name]
        else:
            continue
        assert fqn in state_dict, f'{fqn} not found in state dict: {state_dict.keys()}'
        original_param = state_dict[fqn]
        dtensor_param = distribute_tensor(original_param, mesh, placement_strategy.output_spec.placements)
        local_param = dtensor_param.to_local()
        state_dict[fqn] = torch.nn.Parameter(local_param) if isinstance(original_param, torch.nn.Parameter) else local_param
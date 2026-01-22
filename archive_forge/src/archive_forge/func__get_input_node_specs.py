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
def _get_input_node_specs(node: Node, placement_strategies: Dict[Node, PlacementStrategy]) -> Tuple[DTensorSpec, ...]:
    """
    Get the input specs of a node.
    """
    input_specs_list: List[DTensorSpec] = []
    for input_arg in node.all_input_nodes:
        if input_arg in placement_strategies:
            input_specs_list.append(placement_strategies[input_arg].output_spec)
        else:
            raise ValueError(f'{input_arg} does not have output_spec populated.')
    return tuple(input_specs_list)
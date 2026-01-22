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
def _insert_reshard_gm(gm: torch.fx.GraphModule, node: Node, input_arg: Node, input_arg_spec: DTensorSpec, desired_spec: DTensorSpec) -> None:
    """
    Transform the graph for tensor redistribution.
    """
    input_arg_spec.tensor_meta = input_arg.meta['tensor_meta']
    desired_spec.tensor_meta = input_arg.meta['tensor_meta']
    input_arg_tensor = input_arg.meta['val']

    def reshard_fn(local_tensor: torch.Tensor) -> torch.Tensor:
        return redistribute_local_tensor(local_tensor, input_arg_spec, desired_spec)
    reshard_gm = make_fx(reshard_fn)(input_arg_tensor)
    reshard_gm_nodes = list(reshard_gm.graph.nodes)
    input_node = reshard_gm_nodes[0]
    with gm.graph.inserting_before(node):
        output_node = gm.graph.graph_copy(reshard_gm.graph, val_map={input_node: input_arg})
    node.replace_input_with(input_arg, output_node)
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
def _mark_sharding(gm: GraphModule, graph_signature: ExportGraphSignature, mesh: DeviceMesh, parameter_placements: Dict[str, Placement]) -> Dict[Node, PlacementStrategy]:
    """
    Mark the sharding strategy for each node in the graph module.
    """
    placement_strategies: Dict[Node, PlacementStrategy] = _mark_tensor_parallel_shardings(gm, graph_signature, mesh, parameter_placements)
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            if node not in placement_strategies:
                placement_strategies[node] = _create_placement_strategy(node, mesh, placements=(Replicate(),))
            node.meta['sharding'] = placement_strategies[node]
        elif node.op == 'call_function':
            if node.target == operator.getitem:
                input_nodes = node.all_input_nodes
                assert len(input_nodes) == 1, f'non-compute op only support one input now, found node: {node} with length of inputs: {len(node.args)}'
                arg_strategy = placement_strategies[input_nodes[0]]
                placement_strategies[node] = _create_placement_strategy(node, mesh, placements=arg_strategy.output_spec.placements, input_specs=_get_input_node_specs(node, placement_strategies))
                node.meta['sharding'] = placement_strategies[node]
            else:
                op_schema = _get_op_schema(node, placement_strategies)
                if op_schema.op not in DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs and op_schema.op not in DTensor._op_dispatcher.sharding_propagator.op_to_rules:
                    output_sharding = _generate_default_output_sharding(node, mesh, op_schema)
                else:
                    output_sharding = DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding(op_schema)
                placement_strategies[node] = PlacementStrategy(output_spec=_get_output_spec_from_output_sharding(output_sharding), input_specs=output_sharding.schema_suggestions[0].args_spec if output_sharding.schema_suggestions is not None else _get_input_node_specs(node, placement_strategies))
                node.meta['sharding'] = placement_strategies[node]
        elif node.op == 'output':
            node.meta['sharding'] = None
        else:
            raise RuntimeError(f'op code {node.op} not supported')
    return placement_strategies
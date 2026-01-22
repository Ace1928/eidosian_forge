from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule(aten.stack.default)
def _prop_stack(op_schema: OpSchema) -> OutputSharding:
    tensors = op_schema.args_schema[0]
    dim = 0 if len(op_schema.args_schema) == 1 else cast(int, op_schema.args_schema[1])
    assert isinstance(tensors, list) and len(tensors) > 0, 'expect at least one tensor to stack'
    assert all((isinstance(t, DTensorSpec) for t in tensors)), f'expect a list of DTensorSpecs, but got {tensors}'
    assert all((t.shape == tensors[0].shape for t in tensors)), f'expect all tensors to have the same shape, but got {tensors}.'
    assert all((t.placements == tensors[0].placements for t in tensors)), f'expect all tensors to have the same placements, but got {tensors}.'
    assert all((not p.is_shard(dim) for p in tensors[0].placements)), 'DTensor does not support stack on sharded dimension.'
    return OutputSharding(output_spec=DTensorSpec(mesh=tensors[0].mesh, placements=tensors[0].placements))
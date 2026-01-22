from typing import cast, Dict, List, Optional, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import prod
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
def _gen_reshard_suggestions(op_schema: OpSchema, input_dims: List[str], input_specs: Tuple[DTensorSpec, ...], dim_to_sharding: Dict[str, int], pending_sum: List[int]) -> OutputSharding:
    suggested_arg_specs: List[DTensorSpec] = []
    for input_dim, input_spec in zip(input_dims, input_specs):
        dim_map = [dim_to_sharding[dim] for dim in input_dim]
        suggested_arg_specs.append(DTensorSpec.from_dim_map(mesh=input_spec.mesh, dim_map=dim_map, sums=pending_sum, tensor_meta=input_spec.tensor_meta))
    suggested_schema = OpSchema(op_schema.op, tuple(suggested_arg_specs), {})
    suggested_schema._inplace_rewrap_schema_suggestion(op_schema)
    return OutputSharding(None, schema_suggestions=[suggested_schema], failed_reason='Input placements op sharding propagation failed, need to reshard!')
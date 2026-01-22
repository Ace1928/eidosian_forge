from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule([aten._fused_adam.default])
def _prop__fused_adam(op_schema: OpSchema):
    NT = 5
    tesnor_list_args: Tuple[List[DTensorSpec]] = op_schema.args_schema[:NT]
    assert all((isinstance(schema, list) for schema in tesnor_list_args))
    assert all((isinstance(s, DTensorSpec) for schema in tesnor_list_args for s in schema))
    tensor_schemas: Tuple[List[DTensorSpec]] = [schema for schema in tesnor_list_args if len(schema)]
    assert all((len(s) == len(tensor_schemas[0]) for s in tensor_schemas)), f'expect the same number of gradients and states, but got {[len(s) for s in tensor_schemas]}.'
    if any((any((t != ts[0] for t in ts)) for ts in zip(*tensor_schemas))):
        new_schemas: Tuple[List[DTensorSpec]] = tuple((op_schema.args_schema[0] if len(s) else s for s in tesnor_list_args))
        return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=new_schemas + op_schema.args_schema[NT:], kwargs_schema=op_schema.kwargs_schema)])
    else:
        return OutputSharding(output_spec=(op_schema.args_schema[0],) * NT)
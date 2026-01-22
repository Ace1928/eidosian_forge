from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule([aten._foreach_add.List, aten._foreach_div.List, aten._foreach_mul.List])
def _prop__foreach_binop_list(op_schema: OpSchema) -> OutputSharding:
    self, other = op_schema.args_schema[:2]
    scalar = None if len(op_schema.args_schema) < 3 else op_schema.args_schema[2]
    assert isinstance(self, list) and all((isinstance(s, DTensorSpec) for s in self)), f'Expect a List[DTensorSpec] but got {self}'
    assert isinstance(other, list) and all((isinstance(o, DTensorSpec) for o in other)), f'Expect a List[DTensorSpec] but got {other}'
    assert len(self) == len(other), f'Two tensor lists must match in length, but got {len(self)} and {len(other)}'
    if any((s != o for s, o in zip(self, other))):
        return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(self, self, scalar) if scalar else (self, self), kwargs_schema=op_schema.kwargs_schema)])
    else:
        return OutputSharding(output_spec=self)
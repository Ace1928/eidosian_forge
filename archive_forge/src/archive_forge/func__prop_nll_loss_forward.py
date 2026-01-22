from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule(aten.nll_loss_forward.default)
def _prop_nll_loss_forward(op_schema: OpSchema) -> OutputSharding:
    self, target = op_schema.args_schema[:2]
    assert isinstance(self, DTensorSpec)
    assert isinstance(target, DTensorSpec)
    if self.placements != target.placements:
        new_self = DTensorSpec(mesh=self.mesh, placements=target.placements, tensor_meta=self.tensor_meta)
        return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(new_self, target) + op_schema.args_schema[2:], kwargs_schema=op_schema.kwargs_schema)])
    else:
        return OutputSharding(output_spec=(DTensorSpec(mesh=self.mesh, placements=(_Partial(),)), DTensorSpec(mesh=self.mesh, placements=(Replicate(),))))
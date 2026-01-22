from typing import cast, List, Optional, Sequence, Tuple
import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def _infer_reduce_dims_map(reduction_dims: List[int], input_ndim: int, keep_dim=False) -> List[int]:
    reduction_dims_map = []
    new_dim_count = 0
    for input_dim in range(input_ndim):
        if input_dim in reduction_dims and (not keep_dim):
            reduction_dims_map.append(-1)
        else:
            reduction_dims_map.append(new_dim_count)
            new_dim_count += 1
    return reduction_dims_map
import math
from functools import wraps
from typing import Callable, Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _make_inplace
def _nll_loss_nd(input: TensorLikeType, target: TensorLikeType, weight: Optional[TensorLikeType], reduction: str, ignore_index: int) -> TensorLikeType:
    torch._check(input.ndim > 0 and input.ndim <= 3, lambda: f'Expected input dimension to be either [1, 2, 3] but received {input.ndim}.')
    torch._check(input.ndim == 1 or input.shape[0] == target.shape[0], lambda: f'Expected input batch size {input.shape[0]} to match target batch size {target.shape[0]}.')
    _check_reduction_value(reduction)
    flat_target = torch.flatten(target)
    ignore_classes_mask = torch.eq(flat_target, ignore_index)
    '\n    from torch._subclasses.fake_tensor import FakeTensor\n    num_classes = input.shape[1] if input.ndim > 1 else input.shape[0]\n    valid_classes_mask = torch.logical_and(\n        (flat_target >= 0), (flat_target < num_classes)\n    )\n    class_check = torch.all(torch.logical_or(ignore_classes_mask, valid_classes_mask))\n    torch._check(\n        isinstance(target, FakeTensor) or bool(class_check.item()),\n        lambda: "A target class is out-of-bounds and not the ignore index.",\n    )\n    '
    ignore_class_weight = torch.scalar_tensor(0, dtype=input.dtype, device=input.device)
    class_weight = torch.scalar_tensor(1, dtype=input.dtype, device=input.device) if weight is None else weight[flat_target]
    current_weight = torch.where(ignore_classes_mask, ignore_class_weight, class_weight)
    if input.ndim == 1:
        loss = -input[target] * current_weight
    elif input.ndim == 2:
        batch_size = input.shape[0]
        loss = -input[torch.arange(batch_size), target] * current_weight
    else:
        batch_size = input.shape[0]
        extent = input.shape[2]
        numel = batch_size * extent
        indices = torch.arange(numel)
        bdx = indices // extent
        kdx = indices % extent
        loss = -input[bdx, flat_target, kdx] * current_weight
    loss = torch.reshape(loss, target.shape)
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        return torch.sum(loss) / torch.sum(current_weight)
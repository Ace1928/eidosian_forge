from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
@register_dispatch_func([torch.ops.aten.detach, torch.ops.aten.clone])
def _apply_fn_on_data(func, *args, **kwargs):
    return MaskedTensor(func(_get_data(args[0])), _maybe_get_mask(args[0]))
from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
@register_function_func([torch.Tensor.where, torch.where])
def _function_where(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, '__torch_function__, torch.where', len_args=3, len_kwargs=0)
    return _MaskedWhere.apply(*args)
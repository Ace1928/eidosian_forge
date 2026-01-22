from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
@register_dispatch_func(NATIVE_BINARY_FNS + NATIVE_INPLACE_BINARY_FNS)
def _general_binary(func, *args, **kwargs):
    return _apply_native_binary(func, *args, **kwargs)
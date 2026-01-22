import torch
from .core import _map_mt_args_kwargs, _masks_match, _tensors_match, _wrap_result, is_masked_tensor
def _is_native_binary(fn):
    return fn in NATIVE_BINARY_FNS or fn in NATIVE_INPLACE_BINARY_FNS
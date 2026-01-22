import torch
from .core import _map_mt_args_kwargs, _wrap_result
def _apply_pass_through_fn(fn, *args, **kwargs):
    data_args, data_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_data())
    result_data = fn(*data_args, **data_kwargs)
    mask_args, mask_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_mask())
    result_mask = fn(*mask_args, **mask_kwargs)
    return _wrap_result(result_data, result_mask)
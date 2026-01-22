import torch
from .core import _map_mt_args_kwargs, _wrap_result
def _is_pass_through_fn(fn):
    return fn in PASSTHROUGH_FNS
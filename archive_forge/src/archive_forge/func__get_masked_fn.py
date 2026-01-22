import warnings
import torch
from .core import is_masked_tensor
from .creation import as_masked_tensor, masked_tensor
def _get_masked_fn(fn):
    if fn == 'all':
        return _masked_all
    return getattr(torch.masked, fn)
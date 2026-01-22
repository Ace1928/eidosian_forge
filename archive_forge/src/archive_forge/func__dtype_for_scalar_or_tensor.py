from collections import namedtuple
import torch
from . import _casting_dicts as _cd
def _dtype_for_scalar_or_tensor(x):
    return x.dtype if isinstance(x, torch.Tensor) else _dtype_for_scalar(type(x))
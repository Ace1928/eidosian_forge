import warnings
import torch
from torch.overrides import get_default_nowrap_functions
def _maybe_get_mask(a):
    if is_masked_tensor(a):
        return a.get_mask()
    return None
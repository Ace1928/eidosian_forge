import warnings
import torch
from torch.overrides import get_default_nowrap_functions
def _masks_match(a, b):
    if is_masked_tensor(a) and is_masked_tensor(b):
        mask_a = a.get_mask()
        mask_b = b.get_mask()
        return _tensors_match(mask_a, mask_b, exact=True)
    return True
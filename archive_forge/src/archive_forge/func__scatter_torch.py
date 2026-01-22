from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _scatter_torch(indices, tensor, new_dimensions):
    import torch
    new_tensor = torch.zeros(new_dimensions, dtype=tensor.dtype, device=tensor.device)
    new_tensor[indices] = tensor
    return new_tensor
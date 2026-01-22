from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _round_torch(tensor, decimals=0):
    """Implement a Torch version of np.round"""
    torch = _i('torch')
    tol = 10 ** decimals
    return torch.round(tensor * tol) / tol
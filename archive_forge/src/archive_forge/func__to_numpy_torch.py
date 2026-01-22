from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _to_numpy_torch(x):
    if getattr(x, 'is_conj', False) and x.is_conj():
        x = x.resolve_conj()
    return x.detach().cpu().numpy()
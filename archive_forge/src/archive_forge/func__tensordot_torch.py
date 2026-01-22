from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _tensordot_torch(tensor1, tensor2, axes):
    torch = _i('torch')
    if not semantic_version.match('>=1.10.0', torch.__version__) and axes == 0:
        return torch.outer(tensor1, tensor2)
    return torch.tensordot(tensor1, tensor2, axes)
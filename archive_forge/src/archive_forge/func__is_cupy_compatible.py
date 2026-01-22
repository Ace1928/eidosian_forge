import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
@staticmethod
def _is_cupy_compatible(arg):
    """
        Returns False if CuPy's functions never accept the arguments as
        parameters due to the following reasons.
        - The inputs include an object of a NumPy's specific class other than
          `np.ndarray`.
        - The inputs include a dtype which is not supported in CuPy.
        """
    if isinstance(arg, ndarray):
        if not arg._supports_cupy:
            return False
    if isinstance(arg, (tuple, list)):
        return all((_RecursiveAttr._is_cupy_compatible(i) for i in arg))
    if isinstance(arg, dict):
        bools = [_RecursiveAttr._is_cupy_compatible(arg[i]) for i in arg]
        return all(bools)
    return True
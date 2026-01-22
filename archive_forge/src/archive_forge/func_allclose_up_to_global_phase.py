from typing import cast, List, Optional, Sequence, Union, Tuple
import numpy as np
from cirq.linalg import tolerance, transformations
from cirq import value
def allclose_up_to_global_phase(a: np.ndarray, b: np.ndarray, *, rtol: float=1e-05, atol: float=1e-08, equal_nan: bool=False) -> bool:
    """Determines if a ~= b * exp(i t) for some t.

    Args:
        a: A numpy array.
        b: Another numpy array.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        equal_nan: Whether or not NaN entries should be considered equal to
            other NaN entries.
    """
    if a.shape != b.shape:
        return False
    a, b = transformations.match_global_phase(a, b)
    return np.allclose(a=a, b=b, rtol=rtol, atol=atol, equal_nan=equal_nan)
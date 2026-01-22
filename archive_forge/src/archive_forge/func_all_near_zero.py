from typing import Union, Iterable, TYPE_CHECKING
import numpy as np
def all_near_zero(a: 'ArrayLike', *, atol: float=1e-08) -> bool:
    """Checks if the tensor's elements are all near zero.

    Args:
        a: Tensor of elements that could all be near zero.
        atol: Absolute tolerance.
    """
    return bool(np.all(np.less_equal(np.abs(a), atol)))
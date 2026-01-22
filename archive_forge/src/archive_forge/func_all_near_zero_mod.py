from typing import Union, Iterable, TYPE_CHECKING
import numpy as np
def all_near_zero_mod(a: Union[float, complex, Iterable[float], np.ndarray], period: float, *, atol: float=1e-08) -> bool:
    """Checks if the tensor's elements are all near multiples of the period.

    Args:
        a: Tensor of elements that could all be near multiples of the period.
        period: The period, e.g. 2 pi when working in radians.
        atol: Absolute tolerance.
    """
    b = (np.asarray(a) + period / 2) % period - period / 2
    return bool(np.all(np.less_equal(np.abs(b), atol)))
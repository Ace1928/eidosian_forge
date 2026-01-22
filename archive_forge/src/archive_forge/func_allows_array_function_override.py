from numpy.core.overrides import ARRAY_FUNCTIONS as _array_functions
from numpy import ufunc as _ufunc
import numpy.core.umath as _umath
def allows_array_function_override(func):
    """Determine if a Numpy function can be overridden via `__array_function__`

    Parameters
    ----------
    func : callable
        Function that may be overridable via `__array_function__`

    Returns
    -------
    bool
        `True` if `func` is a function in the Numpy API that is
        overridable via `__array_function__` and `False` otherwise.
    """
    return func in _array_functions
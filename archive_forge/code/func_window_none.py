import functools
from numbers import Number
import numpy as np
from matplotlib import _api, _docstring, cbook
def window_none(x):
    """
    No window function; simply return *x*.

    See Also
    --------
    window_hanning : Another window algorithm.
    """
    return x
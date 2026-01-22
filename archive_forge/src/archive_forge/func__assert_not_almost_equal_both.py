import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def _assert_not_almost_equal_both(a, b, **kwargs):
    """
    Check that two objects are not approximately equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    _assert_not_almost_equal(a, b, **kwargs)
    _assert_not_almost_equal(b, a, **kwargs)
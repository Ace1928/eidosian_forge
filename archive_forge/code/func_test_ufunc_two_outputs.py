import numbers
import operator
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
def test_ufunc_two_outputs(self):
    mantissa, exponent = np.frexp(2 ** (-3))
    expected = (ArrayLike(mantissa), ArrayLike(exponent))
    _assert_equal_type_and_value(np.frexp(ArrayLike(2 ** (-3))), expected)
    _assert_equal_type_and_value(np.frexp(ArrayLike(np.array(2 ** (-3)))), expected)
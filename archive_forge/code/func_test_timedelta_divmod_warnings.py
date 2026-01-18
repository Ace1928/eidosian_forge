import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.skipif(IS_WASM, reason='does not work in wasm')
@pytest.mark.parametrize('op1, op2', [(np.timedelta64(10, 'us'), np.timedelta64(0, 'us')), (np.timedelta64('NaT'), np.timedelta64(50, 'us')), (np.timedelta64(np.iinfo(np.int64).min), np.timedelta64(-1))])
def test_timedelta_divmod_warnings(self, op1, op2):
    with assert_warns(RuntimeWarning):
        expected = (op1 // op2, op1 % op2)
    with assert_warns(RuntimeWarning):
        actual = divmod(op1, op2)
    assert_equal(actual, expected)
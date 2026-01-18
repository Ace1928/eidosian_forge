import numpy as np
from itertools import product
from numpy.testing import assert_equal, assert_allclose
from pytest import raises as assert_raises
import pytest
from scipy.signal import upfirdn, firwin
from scipy.signal._upfirdn import _output_len, _upfirdn_modes
from scipy.signal._upfirdn_apply import _pad_test
@pytest.mark.parametrize('x_dtype', _UPFIRDN_TYPES)
@pytest.mark.parametrize('h_dtype', _UPFIRDN_TYPES)
@pytest.mark.parametrize('p_max, q_max', list(product((10, 100), (10, 100))))
def test_vs_naive(self, x_dtype, h_dtype, p_max, q_max):
    tests = self._random_factors(p_max, q_max, h_dtype, x_dtype)
    for test in tests:
        test()
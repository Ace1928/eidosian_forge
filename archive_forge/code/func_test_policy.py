from multiprocessing import Pool
from multiprocessing.pool import Pool as PWL
import re
import math
from fractions import Fraction
import numpy as np
from numpy.testing import assert_equal, assert_
import pytest
from pytest import raises as assert_raises
import hypothesis.extra.numpy as npst
from hypothesis import given, strategies, reproduce_failure  # noqa: F401
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import xp_assert_equal
from scipy._lib._util import (_aligned_zeros, check_random_state, MapWrapper,
def test_policy(self):
    data = np.array([1, 2, 3, np.nan])
    contains_nan, nan_policy = _contains_nan(data, nan_policy='propagate')
    assert contains_nan
    assert nan_policy == 'propagate'
    contains_nan, nan_policy = _contains_nan(data, nan_policy='omit')
    assert contains_nan
    assert nan_policy == 'omit'
    msg = 'The input contains nan values'
    with pytest.raises(ValueError, match=msg):
        _contains_nan(data, nan_policy='raise')
    msg = 'nan_policy must be one of'
    with pytest.raises(ValueError, match=msg):
        _contains_nan(data, nan_policy='nan')
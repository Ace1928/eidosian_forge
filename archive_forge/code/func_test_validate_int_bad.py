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
@pytest.mark.parametrize('n', [4.0, np.array([4]), Fraction(4, 1)])
def test_validate_int_bad(self, n):
    with pytest.raises(TypeError, match='n must be an integer'):
        _validate_int(n, 'n')
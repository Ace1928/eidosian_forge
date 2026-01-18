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
def test_old_keyword_deprecated(self):
    dep_msg = 'Use of keyword argument `old` is deprecated'
    res1 = self.old_keyword_deprecated(10)
    res2 = self.old_keyword_deprecated(new=10)
    with pytest.warns(DeprecationWarning, match=dep_msg):
        res3 = self.old_keyword_deprecated(old=10)
    assert res1 == res2 == res3 == 10
    message = re.escape('old_keyword_deprecated() got an unexpected')
    with pytest.raises(TypeError, match=message):
        self.old_keyword_deprecated(unexpected=10)
    message = re.escape('old_keyword_deprecated() got multiple')
    with pytest.raises(TypeError, match=message):
        self.old_keyword_deprecated(10, new=10)
    with pytest.raises(TypeError, match=message), pytest.warns(DeprecationWarning, match=dep_msg):
        self.old_keyword_deprecated(10, old=10)
    with pytest.raises(TypeError, match=message), pytest.warns(DeprecationWarning, match=dep_msg):
        self.old_keyword_deprecated(new=10, old=10)
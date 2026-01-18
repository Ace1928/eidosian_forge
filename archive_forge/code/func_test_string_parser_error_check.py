import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_string_parser_error_check(self):
    assert_raises(ValueError, np.array, ['badvalue'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980X'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-00'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-13'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-1'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-1-02'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-Mor'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-01-'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-01-0'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-01-00'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-01-32'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1979-02-29'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-30'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-03-32'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-04-31'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-05-32'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-06-31'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-07-32'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-08-32'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-09-31'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-10-32'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-11-31'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-12-32'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03%'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03 q'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03 25'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03T25'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03 24:01'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03T24:01'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03 -1'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03 01:'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03 01:-1'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03 01:60'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03 01:60:'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03 01:10:-1'], np.dtype('M8[us]'))
    assert_raises(ValueError, np.array, ['1980-02-03 01:01:60'], np.dtype('M8[us]'))
    with assert_warns(DeprecationWarning):
        assert_raises(ValueError, np.array, ['1980-02-03 01:01:00+0661'], np.dtype('M8[us]'))
    with assert_warns(DeprecationWarning):
        assert_raises(ValueError, np.array, ['1980-02-03 01:01:00+2500'], np.dtype('M8[us]'))
    with assert_warns(DeprecationWarning):
        assert_raises(ValueError, np.array, ['1980-02-03 01:01:00-0070'], np.dtype('M8[us]'))
    with assert_warns(DeprecationWarning):
        assert_raises(ValueError, np.array, ['1980-02-03 01:01:00-3000'], np.dtype('M8[us]'))
    with assert_warns(DeprecationWarning):
        assert_raises(ValueError, np.array, ['1980-02-03 01:01:00-25:00'], np.dtype('M8[us]'))
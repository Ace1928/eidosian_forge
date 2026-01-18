import os
from platform import machine
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import (
from ..testing import suppress_warnings
def test_ulp():
    assert ulp() == np.finfo(np.float64).eps
    assert ulp(1.0) == np.finfo(np.float64).eps
    assert ulp(np.float32(1.0)) == np.finfo(np.float32).eps
    assert ulp(np.float32(1.999)) == np.finfo(np.float32).eps
    assert ulp(1) == 1
    assert ulp(2 ** 63 - 1) == 1
    assert ulp(-1) == 1
    assert ulp(7.999) == ulp(4.0)
    assert ulp(-7.999) == ulp(4.0)
    assert ulp(np.float64(2 ** 54 - 2)) == 2
    assert ulp(np.float64(2 ** 54)) == 4
    assert ulp(np.float64(2 ** 54)) == 4
    assert np.isnan(ulp(np.inf))
    assert np.isnan(ulp(-np.inf))
    assert np.isnan(ulp(np.nan))
    subn64 = np.float64(2 ** (-1022 - 52))
    subn32 = np.float32(2 ** (-126 - 23))
    assert ulp(0.0) == subn64
    assert ulp(np.float64(0)) == subn64
    assert ulp(np.float32(0)) == subn32
    assert ulp(subn64 * np.float64(2 ** 52)) == subn64
    assert ulp(subn64 * np.float64(2 ** 53)) == subn64 * 2
    assert ulp(subn32 * np.float32(2 ** 23)) == subn32
    assert ulp(subn32 * np.float32(2 ** 24)) == subn32 * 2
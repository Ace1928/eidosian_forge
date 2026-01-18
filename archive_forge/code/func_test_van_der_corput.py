import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
def test_van_der_corput(self):
    sample = van_der_corput(10)
    out = [0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625, 0.5625]
    assert_allclose(sample, out)
    sample = van_der_corput(10, workers=4)
    assert_allclose(sample, out)
    sample = van_der_corput(10, workers=8)
    assert_allclose(sample, out)
    sample = van_der_corput(7, start_index=3)
    assert_allclose(sample, out[3:])
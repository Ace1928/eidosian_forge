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
def test_scale_random(self):
    rng = np.random.default_rng(317589836511269190194010915937762468165)
    sample = rng.random((30, 10))
    a = -rng.random(10) * 10
    b = rng.random(10) * 10
    scaled = qmc.scale(sample, a, b, reverse=False)
    unscaled = qmc.scale(scaled, a, b, reverse=True)
    assert_allclose(unscaled, sample)
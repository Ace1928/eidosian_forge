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
class RandomEngine(qmc.QMCEngine):

    def __init__(self, d, optimization=None, seed=None):
        super().__init__(d=d, optimization=optimization, seed=seed)

    def _random(self, n=1, *, workers=1):
        sample = self.rng.random((n, self.d))
        return sample
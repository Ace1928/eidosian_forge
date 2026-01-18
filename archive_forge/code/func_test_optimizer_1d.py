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
def test_optimizer_1d(self):
    engine = self.engine(d=1, scramble=False)
    sample_ref = engine.random(n=64)
    optimal_ = self.engine(d=1, scramble=False, optimization='random-CD')
    sample_ = optimal_.random(n=64)
    assert_array_equal(sample_ref, sample_)
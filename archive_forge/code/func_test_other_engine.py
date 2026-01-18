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
def test_other_engine(self):
    for d in (0, 1, 2):
        base_engine = qmc.Sobol(d=d, scramble=False)
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(d), engine=base_engine, inv_transform=True)
        samples = engine.random()
        assert_equal(samples.shape, (1, d))
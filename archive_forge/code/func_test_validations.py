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
def test_validations(self):
    message = 'Dimension of `engine` must be consistent'
    with pytest.raises(ValueError, match=message):
        qmc.MultivariateNormalQMC([0], engine=qmc.Sobol(d=2))
    message = 'Dimension of `engine` must be consistent'
    with pytest.raises(ValueError, match=message):
        qmc.MultivariateNormalQMC([0, 0, 0], engine=qmc.Sobol(d=4))
    message = '`engine` must be an instance of...'
    with pytest.raises(ValueError, match=message):
        qmc.MultivariateNormalQMC([0, 0], engine=np.random.default_rng())
    message = 'Covariance matrix not PSD.'
    with pytest.raises(ValueError, match=message):
        qmc.MultivariateNormalQMC([0, 0], [[1, 2], [2, 1]])
    message = 'Covariance matrix is not symmetric.'
    with pytest.raises(ValueError, match=message):
        qmc.MultivariateNormalQMC([0, 0], [[1, 0], [2, 1]])
    message = 'Dimension mismatch between mean and covariance.'
    with pytest.raises(ValueError, match=message):
        qmc.MultivariateNormalQMC([0], [[1, 0], [0, 1]])
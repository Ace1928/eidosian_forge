import pickle
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from .test_continuous_basic import check_distribution_rvs
import numpy
import numpy as np
import scipy.linalg
from scipy.stats._multivariate import (_PSD,
from scipy.stats import (multivariate_normal, multivariate_hypergeom,
from scipy.stats import _covariance, Covariance
from scipy import stats
from scipy.integrate import romb, qmc_quad, tplquad
from scipy.special import multigammaln
from scipy._lib._pep440 import Version
from .common_tests import check_random_state_property
from .data._mvt import _qsimvtv
from unittest.mock import patch
def test_random_state_property():
    scale = np.eye(3)
    scale[0, 1] = 0.5
    scale[1, 0] = 0.5
    dists = [[multivariate_normal, ()], [dirichlet, (np.array([1.0]),)], [wishart, (10, scale)], [invwishart, (10, scale)], [multinomial, (5, [0.5, 0.4, 0.1])], [ortho_group, (2,)], [special_ortho_group, (2,)]]
    for distfn, args in dists:
        check_random_state_property(distfn, args)
        check_pickling(distfn, args)
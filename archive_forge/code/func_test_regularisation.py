import copy
import itertools
import re
import sys
import warnings
from io import StringIO
from unittest.mock import Mock
import numpy as np
import pytest
from scipy import linalg, stats
import sklearn
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_spd_matrix
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_regularisation():
    rng = np.random.RandomState(0)
    n_samples, n_features = (10, 5)
    X = np.vstack((np.ones((n_samples // 2, n_features)), np.zeros((n_samples // 2, n_features))))
    for covar_type in COVARIANCE_TYPE:
        gmm = GaussianMixture(n_components=n_samples, reg_covar=0, covariance_type=covar_type, random_state=rng)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            msg = re.escape('Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). Try to decrease the number of components, or increase reg_covar.')
            with pytest.raises(ValueError, match=msg):
                gmm.fit(X)
            gmm.set_params(reg_covar=1e-06).fit(X)
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
def _generate_data(seed, n_samples, n_features, n_components):
    """Randomly generate samples and responsibilities."""
    rs = np.random.RandomState(seed)
    X = rs.random_sample((n_samples, n_features))
    resp = rs.random_sample((n_samples, n_components))
    resp /= resp.sum(axis=1)[:, np.newaxis]
    return (X, resp)
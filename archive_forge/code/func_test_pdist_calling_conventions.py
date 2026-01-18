import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def test_pdist_calling_conventions(self, metric):
    for eo_name in self.rnd_eo_names:
        X = eo[eo_name][::5, ::2]
        if verbose > 2:
            print('testing: ', metric, ' with: ', eo_name)
        if metric in {'dice', 'yule', 'matching', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'kulczynski1'} and 'bool' not in eo_name:
            continue
        self._check_calling_conventions(X, metric)
        if metric == 'seuclidean':
            V = np.var(X.astype(np.float64), axis=0, ddof=1)
            self._check_calling_conventions(X, metric, V=V)
        elif metric == 'mahalanobis':
            V = np.atleast_2d(np.cov(X.astype(np.float64).T))
            VI = np.array(np.linalg.inv(V).T)
            self._check_calling_conventions(X, metric, VI=VI)
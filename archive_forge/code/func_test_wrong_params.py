import re
import sys
from io import StringIO
import numpy as np
import pytest
from scipy import sparse as sp
from sklearn.base import clone
from sklearn.cluster import KMeans, MiniBatchKMeans, k_means, kmeans_plusplus
from sklearn.cluster._k_means_common import (
from sklearn.cluster._kmeans import _labels_inertia, _mini_batch_step
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils._testing import (
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS, threadpool_limits
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
@pytest.mark.parametrize('param, match', [({'n_clusters': n_samples + 1}, 'n_samples.* should be >= n_clusters'), ({'init': X[:2]}, 'The shape of the initial centers .* does not match the number of clusters'), ({'init': lambda X_, k, random_state: X_[:2]}, 'The shape of the initial centers .* does not match the number of clusters'), ({'init': X[:8, :2]}, 'The shape of the initial centers .* does not match the number of features of the data'), ({'init': lambda X_, k, random_state: X_[:8, :2]}, 'The shape of the initial centers .* does not match the number of features of the data')])
def test_wrong_params(Estimator, param, match):
    km = Estimator(n_init=1)
    with pytest.raises(ValueError, match=match):
        km.set_params(**param).fit(X)
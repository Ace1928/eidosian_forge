import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._unsupervised import _silhouette_reduce
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
def test_silhouette_paper_example():
    lower = [5.58, 7.0, 6.5, 7.08, 7.0, 3.83, 4.83, 5.08, 8.17, 5.83, 2.17, 5.75, 6.67, 6.92, 4.92, 6.42, 5.0, 5.58, 6.0, 4.67, 6.42, 3.42, 5.5, 6.42, 6.42, 5.0, 3.92, 6.17, 2.5, 4.92, 6.25, 7.33, 4.5, 2.25, 6.33, 2.75, 6.08, 6.67, 4.25, 2.67, 6.0, 6.17, 6.17, 6.92, 6.17, 5.25, 6.83, 4.5, 3.75, 5.75, 5.42, 6.08, 5.83, 6.67, 3.67, 4.75, 3.0, 6.08, 6.67, 5.0, 5.58, 4.83, 6.17, 5.67, 6.5, 6.92]
    D = np.zeros((12, 12))
    D[np.tril_indices(12, -1)] = lower
    D += D.T
    names = ['BEL', 'BRA', 'CHI', 'CUB', 'EGY', 'FRA', 'IND', 'ISR', 'USA', 'USS', 'YUG', 'ZAI']
    labels1 = [1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1]
    expected1 = {'USA': 0.43, 'BEL': 0.39, 'FRA': 0.35, 'ISR': 0.3, 'BRA': 0.22, 'EGY': 0.2, 'ZAI': 0.19, 'CUB': 0.4, 'USS': 0.34, 'CHI': 0.33, 'YUG': 0.26, 'IND': -0.04}
    score1 = 0.28
    labels2 = [1, 2, 3, 3, 1, 1, 2, 1, 1, 3, 3, 2]
    expected2 = {'USA': 0.47, 'FRA': 0.44, 'BEL': 0.42, 'ISR': 0.37, 'EGY': 0.02, 'ZAI': 0.28, 'BRA': 0.25, 'IND': 0.17, 'CUB': 0.48, 'USS': 0.44, 'YUG': 0.31, 'CHI': 0.31}
    score2 = 0.33
    for labels, expected, score in [(labels1, expected1, score1), (labels2, expected2, score2)]:
        expected = [expected[name] for name in names]
        pytest.approx(expected, silhouette_samples(D, np.array(labels), metric='precomputed'), abs=0.01)
        pytest.approx(score, silhouette_score(D, np.array(labels), metric='precomputed'), abs=0.01)
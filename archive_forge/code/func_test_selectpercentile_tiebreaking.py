import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_selectpercentile_tiebreaking():
    Xs = [[0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0]]
    y = [1]
    dummy_score = lambda X, y: (X[0], X[0])
    for X in Xs:
        sel = SelectPercentile(dummy_score, percentile=34)
        X1 = ignore_warnings(sel.fit_transform)([X], y)
        assert X1.shape[1] == 1
        assert_best_scores_kept(sel)
        sel = SelectPercentile(dummy_score, percentile=67)
        X2 = ignore_warnings(sel.fit_transform)([X], y)
        assert X2.shape[1] == 2
        assert_best_scores_kept(sel)
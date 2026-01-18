import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
def test_pipeline_with_nearest_neighbors_transformer():
    rng = np.random.RandomState(0)
    X = 2 * rng.rand(40, 5) - 1
    X2 = 2 * rng.rand(40, 5) - 1
    y = rng.rand(40, 1)
    n_neighbors = 12
    radius = 1.5
    factor = 2
    k_trans = neighbors.KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance')
    k_trans_factor = neighbors.KNeighborsTransformer(n_neighbors=int(n_neighbors * factor), mode='distance')
    r_trans = neighbors.RadiusNeighborsTransformer(radius=radius, mode='distance')
    r_trans_factor = neighbors.RadiusNeighborsTransformer(radius=int(radius * factor), mode='distance')
    k_reg = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)
    r_reg = neighbors.RadiusNeighborsRegressor(radius=radius)
    test_list = [(k_trans, k_reg), (k_trans_factor, r_reg), (r_trans, r_reg), (r_trans_factor, k_reg)]
    for trans, reg in test_list:
        reg_compact = clone(reg)
        reg_precomp = clone(reg)
        reg_precomp.set_params(metric='precomputed')
        reg_chain = make_pipeline(clone(trans), reg_precomp)
        y_pred_chain = reg_chain.fit(X, y).predict(X2)
        y_pred_compact = reg_compact.fit(X, y).predict(X2)
        assert_allclose(y_pred_chain, y_pred_compact)
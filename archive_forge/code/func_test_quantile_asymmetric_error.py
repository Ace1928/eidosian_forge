import copyreg
import io
import pickle
import re
import warnings
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
from joblib.numpy_pickle import NumpyPickler
from numpy.testing import assert_allclose, assert_array_equal
import sklearn
from sklearn._loss.loss import (
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_regressor
from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification, make_low_rank_matrix, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import get_scorer, mean_gamma_deviance, mean_poisson_deviance
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder
from sklearn.utils import _IS_32BIT, shuffle
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import _convert_container
@pytest.mark.parametrize('quantile', [0.2, 0.5, 0.8])
def test_quantile_asymmetric_error(quantile):
    """Test quantile regression for asymmetric distributed targets."""
    n_samples = 10000
    rng = np.random.RandomState(42)
    X = np.concatenate((np.abs(rng.randn(n_samples)[:, None]), -rng.randint(2, size=(n_samples, 1))), axis=1)
    intercept = 1.23
    coef = np.array([0.5, -2])
    y = rng.exponential(scale=-(X @ coef + intercept) / np.log(1 - quantile), size=n_samples)
    model = HistGradientBoostingRegressor(loss='quantile', quantile=quantile, max_iter=25, random_state=0, max_leaf_nodes=10).fit(X, y)
    assert_allclose(np.mean(model.predict(X) > y), quantile, rtol=0.01)
    pinball_loss = PinballLoss(quantile=quantile)
    loss_true_quantile = pinball_loss(y, X @ coef + intercept)
    loss_pred_quantile = pinball_loss(y, model.predict(X))
    assert loss_pred_quantile <= loss_true_quantile
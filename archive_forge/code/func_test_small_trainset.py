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
def test_small_trainset():
    n_samples = 20000
    original_distrib = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples).reshape(n_samples, 1)
    y = [[class_] * int(prop * n_samples) for class_, prop in original_distrib.items()]
    y = shuffle(np.concatenate(y))
    gb = HistGradientBoostingClassifier()
    X_small, y_small, *_ = gb._get_small_trainset(X, y, seed=42, sample_weight_train=None)
    unique, counts = np.unique(y_small, return_counts=True)
    small_distrib = {class_: count / 10000 for class_, count in zip(unique, counts)}
    assert X_small.shape[0] == 10000
    assert y_small.shape[0] == 10000
    assert small_distrib == pytest.approx(original_distrib)
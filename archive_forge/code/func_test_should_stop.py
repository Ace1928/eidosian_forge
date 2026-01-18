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
@pytest.mark.parametrize('scores, n_iter_no_change, tol, stopping', [([], 1, 0.001, False), ([1, 1, 1], 5, 0.001, False), ([1, 1, 1, 1, 1], 5, 0.001, False), ([1, 2, 3, 4, 5, 6], 5, 0.001, False), ([1, 2, 3, 4, 5, 6], 5, 0.0, False), ([1, 2, 3, 4, 5, 6], 5, 0.999, False), ([1, 2, 3, 4, 5, 6], 5, 5 - 1e-05, False), ([1] * 6, 5, 0.0, True), ([1] * 6, 5, 0.001, True), ([1] * 6, 5, 5, True)])
def test_should_stop(scores, n_iter_no_change, tol, stopping):
    gbdt = HistGradientBoostingClassifier(n_iter_no_change=n_iter_no_change, tol=tol)
    assert gbdt._should_stop(scores) == stopping
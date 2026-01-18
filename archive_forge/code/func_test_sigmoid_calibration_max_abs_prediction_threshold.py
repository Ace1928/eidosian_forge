import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import (
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
def test_sigmoid_calibration_max_abs_prediction_threshold(global_random_seed):
    random_state = np.random.RandomState(seed=global_random_seed)
    n = 100
    y = random_state.randint(0, 2, size=n)
    predictions_small = random_state.uniform(low=-2, high=2, size=100)
    threshold_1 = 0.1
    a1, b1 = _sigmoid_calibration(predictions=predictions_small, y=y, max_abs_prediction_threshold=threshold_1)
    threshold_2 = 10
    a2, b2 = _sigmoid_calibration(predictions=predictions_small, y=y, max_abs_prediction_threshold=threshold_2)
    a3, b3 = _sigmoid_calibration(predictions=predictions_small, y=y)
    atol = 1e-06
    assert_allclose(a1, a2, atol=atol)
    assert_allclose(a2, a3, atol=atol)
    assert_allclose(b1, b2, atol=atol)
    assert_allclose(b2, b3, atol=atol)
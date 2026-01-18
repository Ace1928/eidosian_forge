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
@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
@pytest.mark.parametrize('ensemble', [True, False])
def test_parallel_execution(data, method, ensemble):
    """Test parallel calibration"""
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = make_pipeline(StandardScaler(), LinearSVC(dual='auto', random_state=42))
    cal_clf_parallel = CalibratedClassifierCV(estimator, method=method, n_jobs=2, ensemble=ensemble)
    cal_clf_parallel.fit(X_train, y_train)
    probs_parallel = cal_clf_parallel.predict_proba(X_test)
    cal_clf_sequential = CalibratedClassifierCV(estimator, method=method, n_jobs=1, ensemble=ensemble)
    cal_clf_sequential.fit(X_train, y_train)
    probs_sequential = cal_clf_sequential.predict_proba(X_test)
    assert_allclose(probs_parallel, probs_sequential)
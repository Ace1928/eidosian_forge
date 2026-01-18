import pickle
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import datasets, linear_model, metrics
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.linear_model import _stochastic_gradient
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.svm import OneClassSVM
from sklearn.utils._testing import (
@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_sgd_predict_proba_method_access(klass):
    for loss in linear_model.SGDClassifier.loss_functions:
        clf = SGDClassifier(loss=loss)
        if loss in ('log_loss', 'modified_huber'):
            assert hasattr(clf, 'predict_proba')
            assert hasattr(clf, 'predict_log_proba')
        else:
            inner_msg = 'probability estimates are not available for loss={!r}'.format(loss)
            assert not hasattr(clf, 'predict_proba')
            assert not hasattr(clf, 'predict_log_proba')
            with pytest.raises(AttributeError, match="has no attribute 'predict_proba'") as exec_info:
                clf.predict_proba
            assert isinstance(exec_info.value.__cause__, AttributeError)
            assert inner_msg in str(exec_info.value.__cause__)
            with pytest.raises(AttributeError, match="has no attribute 'predict_log_proba'") as exec_info:
                clf.predict_log_proba
            assert isinstance(exec_info.value.__cause__, AttributeError)
            assert inner_msg in str(exec_info.value.__cause__)
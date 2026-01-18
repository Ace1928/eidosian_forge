import re
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.metrics._ranking import _dcg_sample_scores, _ndcg_sample_scores
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import (
@pytest.mark.parametrize('drop', [True, False])
def test_precision_recall_curve_toydata(drop):
    with np.errstate(all='raise'):
        y_true = [0, 1]
        y_score = [0, 1]
        p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 1, 1])
        assert_array_almost_equal(r, [1, 1, 0])
        assert_almost_equal(auc_prc, 1.0)
        y_true = [0, 1]
        y_score = [1, 0]
        p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 0.0, 1.0])
        assert_array_almost_equal(r, [1.0, 0.0, 0.0])
        assert_almost_equal(auc_prc, 0.5)
        y_true = [1, 0]
        y_score = [1, 1]
        p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 1])
        assert_array_almost_equal(r, [1.0, 0])
        assert_almost_equal(auc_prc, 0.5)
        y_true = [1, 0]
        y_score = [1, 0]
        p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 1, 1])
        assert_array_almost_equal(r, [1, 1, 0])
        assert_almost_equal(auc_prc, 1.0)
        y_true = [1, 0]
        y_score = [0.5, 0.5]
        p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 1])
        assert_array_almost_equal(r, [1, 0.0])
        assert_almost_equal(auc_prc, 0.5)
        y_true = [0, 0]
        y_score = [0.25, 0.75]
        with pytest.warns(UserWarning, match='No positive class found in y_true'):
            p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        with pytest.warns(UserWarning, match='No positive class found in y_true'):
            auc_prc = average_precision_score(y_true, y_score)
        assert_allclose(p, [0, 0, 1])
        assert_allclose(r, [1, 1, 0])
        assert_allclose(auc_prc, 0)
        y_true = [1, 1]
        y_score = [0.25, 0.75]
        p, r, _ = precision_recall_curve(y_true, y_score, drop_intermediate=drop)
        assert_almost_equal(average_precision_score(y_true, y_score), 1.0)
        assert_array_almost_equal(p, [1.0, 1.0, 1.0])
        assert_array_almost_equal(r, [1, 0.5, 0.0])
        y_true = np.array([[0, 1], [0, 1]])
        y_score = np.array([[0, 1], [0, 1]])
        with pytest.warns(UserWarning, match='No positive class found in y_true'):
            assert_allclose(average_precision_score(y_true, y_score, average='macro'), 0.5)
        with pytest.warns(UserWarning, match='No positive class found in y_true'):
            assert_allclose(average_precision_score(y_true, y_score, average='weighted'), 1.0)
        assert_allclose(average_precision_score(y_true, y_score, average='samples'), 1.0)
        assert_allclose(average_precision_score(y_true, y_score, average='micro'), 1.0)
        y_true = np.array([[0, 1], [0, 1]])
        y_score = np.array([[0, 1], [1, 0]])
        with pytest.warns(UserWarning, match='No positive class found in y_true'):
            assert_allclose(average_precision_score(y_true, y_score, average='macro'), 0.5)
        with pytest.warns(UserWarning, match='No positive class found in y_true'):
            assert_allclose(average_precision_score(y_true, y_score, average='weighted'), 1.0)
        assert_allclose(average_precision_score(y_true, y_score, average='samples'), 0.75)
        assert_allclose(average_precision_score(y_true, y_score, average='micro'), 0.5)
        y_true = np.array([[1, 0], [0, 1]])
        y_score = np.array([[0, 1], [1, 0]])
        assert_almost_equal(average_precision_score(y_true, y_score, average='macro'), 0.5)
        assert_almost_equal(average_precision_score(y_true, y_score, average='weighted'), 0.5)
        assert_almost_equal(average_precision_score(y_true, y_score, average='samples'), 0.5)
        assert_almost_equal(average_precision_score(y_true, y_score, average='micro'), 0.5)
        y_true = np.array([[0, 0], [0, 0]])
        y_score = np.array([[0, 1], [0, 1]])
        with pytest.warns(UserWarning, match='No positive class found in y_true'):
            assert_allclose(average_precision_score(y_true, y_score, average='macro'), 0.0)
        assert_allclose(average_precision_score(y_true, y_score, average='weighted'), 0.0)
        with pytest.warns(UserWarning, match='No positive class found in y_true'):
            assert_allclose(average_precision_score(y_true, y_score, average='samples'), 0.0)
        with pytest.warns(UserWarning, match='No positive class found in y_true'):
            assert_allclose(average_precision_score(y_true, y_score, average='micro'), 0.0)
        y_true = np.array([[1, 1], [1, 1]])
        y_score = np.array([[0, 1], [0, 1]])
        assert_allclose(average_precision_score(y_true, y_score, average='macro'), 1.0)
        assert_allclose(average_precision_score(y_true, y_score, average='weighted'), 1.0)
        assert_allclose(average_precision_score(y_true, y_score, average='samples'), 1.0)
        assert_allclose(average_precision_score(y_true, y_score, average='micro'), 1.0)
        y_true = np.array([[1, 0], [0, 1]])
        y_score = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert_almost_equal(average_precision_score(y_true, y_score, average='macro'), 0.5)
        assert_almost_equal(average_precision_score(y_true, y_score, average='weighted'), 0.5)
        assert_almost_equal(average_precision_score(y_true, y_score, average='samples'), 0.5)
        assert_almost_equal(average_precision_score(y_true, y_score, average='micro'), 0.5)
    with np.errstate(all='ignore'):
        y_true = np.array([[0, 0], [0, 1]])
        y_score = np.array([[0, 0], [0, 1]])
        with pytest.warns(UserWarning, match='No positive class found in y_true'):
            assert_allclose(average_precision_score(y_true, y_score, average='weighted'), 1)
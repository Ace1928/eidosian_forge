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
@pytest.mark.parametrize('y_true, true_score, labels', [(np.array([0, 1, 1, 2]), 0.75, [0, 1, 2, 3]), (np.array([0, 1, 1, 1]), 0.5, [0, 1, 2, 3]), (np.array([1, 1, 1, 1]), 0.5, [0, 1, 2, 3]), (np.array(['a', 'e', 'e', 'a']), 0.75, ['a', 'b', 'd', 'e'])])
@pytest.mark.parametrize('labels_as_ndarray', [True, False])
def test_top_k_accuracy_score_multiclass_with_labels(y_true, true_score, labels, labels_as_ndarray):
    """Test when labels and y_score are multiclass."""
    if labels_as_ndarray:
        labels = np.asarray(labels)
    y_score = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.3, 0.4, 0.2], [0.4, 0.1, 0.2, 0.3], [0.3, 0.2, 0.4, 0.1]])
    score = top_k_accuracy_score(y_true, y_score, k=2, labels=labels)
    assert score == pytest.approx(true_score)
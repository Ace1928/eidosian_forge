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
@pytest.mark.parametrize('msg, y_true, labels', [("Parameter 'labels' must be unique", np.array([0, 1, 2, 2]), [0, 2, 0]), ("Parameter 'labels' must be unique", np.array(['a', 'b', 'c', 'c']), ['a', 'a', 'b']), ("Number of classes in y_true not equal to the number of columns in 'y_score'", np.array([0, 2, 0, 2]), None), ("Parameter 'labels' must be ordered", np.array(['a', 'b', 'c', 'c']), ['a', 'c', 'b']), ("Number of given labels, 2, not equal to the number of columns in 'y_score', 3", np.array([0, 1, 2, 2]), [0, 1]), ("Number of given labels, 2, not equal to the number of columns in 'y_score', 3", np.array(['a', 'b', 'c', 'c']), ['a', 'b']), ("Number of given labels, 4, not equal to the number of columns in 'y_score', 3", np.array([0, 1, 2, 2]), [0, 1, 2, 3]), ("Number of given labels, 4, not equal to the number of columns in 'y_score', 3", np.array(['a', 'b', 'c', 'c']), ['a', 'b', 'c', 'd']), ("'y_true' contains labels not in parameter 'labels'", np.array(['a', 'b', 'c', 'e']), ['a', 'b', 'c']), ("'y_true' contains labels not in parameter 'labels'", np.array(['a', 'b', 'c', 'd']), ['a', 'b', 'c']), ("'y_true' contains labels not in parameter 'labels'", np.array([0, 1, 2, 3]), [0, 1, 2])])
@pytest.mark.parametrize('multi_class', ['ovo', 'ovr'])
def test_roc_auc_score_multiclass_labels_error(msg, y_true, labels, multi_class):
    y_scores = np.array([[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.35, 0.5, 0.15], [0, 0.2, 0.8]])
    with pytest.raises(ValueError, match=msg):
        roc_auc_score(y_true, y_scores, labels=labels, multi_class=multi_class)
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
@pytest.mark.parametrize('y_score, k, true_score', [(np.array([-1, -1, 1, 1]), 1, 1), (np.array([-1, 1, -1, 1]), 1, 0.5), (np.array([-1, 1, -1, 1]), 2, 1), (np.array([0.2, 0.2, 0.7, 0.7]), 1, 1), (np.array([0.2, 0.7, 0.2, 0.7]), 1, 0.5), (np.array([0.2, 0.7, 0.2, 0.7]), 2, 1)])
def test_top_k_accuracy_score_binary(y_score, k, true_score):
    y_true = [0, 0, 1, 1]
    threshold = 0.5 if y_score.min() >= 0 and y_score.max() <= 1 else 0
    y_pred = (y_score > threshold).astype(np.int64) if k == 1 else y_true
    score = top_k_accuracy_score(y_true, y_score, k=k)
    score_acc = accuracy_score(y_true, y_pred)
    assert score == score_acc == pytest.approx(true_score)
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
def _my_lrap(y_true, y_score):
    """Simple implementation of label ranking average precision"""
    check_consistent_length(y_true, y_score)
    y_true = check_array(y_true)
    y_score = check_array(y_score)
    n_samples, n_labels = y_true.shape
    score = np.empty((n_samples,))
    for i in range(n_samples):
        unique_rank, inv_rank = np.unique(y_score[i], return_inverse=True)
        n_ranks = unique_rank.size
        rank = n_ranks - inv_rank
        corr_rank = np.bincount(rank, minlength=n_ranks + 1).cumsum()
        rank = corr_rank[rank]
        relevant = y_true[i].nonzero()[0]
        if relevant.size == 0 or relevant.size == n_labels:
            score[i] = 1
            continue
        score[i] = 0.0
        for label in relevant:
            n_ranked_above = sum((rank[r] <= rank[label] for r in relevant))
            score[i] += n_ranked_above / rank[label]
        score[i] /= relevant.size
    return score.mean()
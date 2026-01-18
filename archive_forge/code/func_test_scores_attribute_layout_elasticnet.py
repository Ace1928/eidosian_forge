import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import (
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
def test_scores_attribute_layout_elasticnet():
    X, y = make_classification(n_samples=1000, random_state=0)
    cv = StratifiedKFold(n_splits=5)
    l1_ratios = [0.1, 0.9]
    Cs = [0.1, 1, 10]
    lrcv = LogisticRegressionCV(penalty='elasticnet', solver='saga', l1_ratios=l1_ratios, Cs=Cs, cv=cv, random_state=0, max_iter=250, tol=0.001)
    lrcv.fit(X, y)
    avg_scores_lrcv = lrcv.scores_[1].mean(axis=0)
    for i, C in enumerate(Cs):
        for j, l1_ratio in enumerate(l1_ratios):
            lr = LogisticRegression(penalty='elasticnet', solver='saga', C=C, l1_ratio=l1_ratio, random_state=0, max_iter=250, tol=0.001)
            avg_score_lr = cross_val_score(lr, X, y, cv=cv).mean()
            assert avg_scores_lrcv[i, j] == pytest.approx(avg_score_lr)
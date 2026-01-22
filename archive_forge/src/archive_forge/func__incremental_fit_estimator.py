import numbers
import time
import warnings
from collections import Counter
from contextlib import suppress
from functools import partial
from numbers import Real
from traceback import format_exc
import numpy as np
import scipy.sparse as sp
from joblib import logger
from ..base import clone, is_classifier
from ..exceptions import FitFailedWarning, UnsetMetadataPassedError
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..preprocessing import LabelEncoder
from ..utils import Bunch, _safe_indexing, check_random_state, indexable
from ..utils._param_validation import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _safe_split
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_method_params, _num_samples
from ._split import check_cv
def _incremental_fit_estimator(estimator, X, y, classes, train, test, train_sizes, scorer, return_times, error_score, fit_params):
    """Train estimator on training subsets incrementally and compute scores."""
    train_scores, test_scores, fit_times, score_times = ([], [], [], [])
    partitions = zip(train_sizes, np.split(train, train_sizes)[:-1])
    if fit_params is None:
        fit_params = {}
    if classes is None:
        partial_fit_func = partial(estimator.partial_fit, **fit_params)
    else:
        partial_fit_func = partial(estimator.partial_fit, classes=classes, **fit_params)
    for n_train_samples, partial_train in partitions:
        train_subset = train[:n_train_samples]
        X_train, y_train = _safe_split(estimator, X, y, train_subset)
        X_partial_train, y_partial_train = _safe_split(estimator, X, y, partial_train)
        X_test, y_test = _safe_split(estimator, X, y, test, train_subset)
        start_fit = time.time()
        if y_partial_train is None:
            partial_fit_func(X_partial_train)
        else:
            partial_fit_func(X_partial_train, y_partial_train)
        fit_time = time.time() - start_fit
        fit_times.append(fit_time)
        start_score = time.time()
        test_scores.append(_score(estimator, X_test, y_test, scorer, score_params=None, error_score=error_score))
        train_scores.append(_score(estimator, X_train, y_train, scorer, score_params=None, error_score=error_score))
        score_time = time.time() - start_score
        score_times.append(score_time)
    ret = (train_scores, test_scores, fit_times, score_times) if return_times else (train_scores, test_scores)
    return np.array(ret).T
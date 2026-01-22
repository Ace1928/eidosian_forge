import pickle
import re
import warnings
from contextlib import nullcontext
from copy import deepcopy
from functools import partial, wraps
from inspect import signature
from numbers import Integral, Real
import joblib
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from .. import config_context
from ..base import (
from ..datasets import (
from ..exceptions import DataConversionWarning, NotFittedError, SkipTestWarning
from ..feature_selection import SelectFromModel, SelectKBest
from ..linear_model import (
from ..metrics import accuracy_score, adjusted_rand_score, f1_score
from ..metrics.pairwise import linear_kernel, pairwise_distances, rbf_kernel
from ..model_selection import ShuffleSplit, train_test_split
from ..model_selection._validation import _safe_split
from ..pipeline import make_pipeline
from ..preprocessing import StandardScaler, scale
from ..random_projection import BaseRandomProjection
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils._array_api import (
from ..utils._array_api import (
from ..utils._param_validation import (
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import check_is_fitted
from . import IS_PYPY, is_scalar_nan, shuffle
from ._param_validation import Interval
from ._tags import (
from ._testing import (
from .validation import _num_samples, has_fit_parameter
def _yield_all_checks(estimator):
    name = estimator.__class__.__name__
    tags = _safe_tags(estimator)
    if '2darray' not in tags['X_types']:
        warnings.warn("Can't test estimator {} which requires input  of type {}".format(name, tags['X_types']), SkipTestWarning)
        return
    if tags['_skip_test']:
        warnings.warn('Explicit SKIP via _skip_test tag for estimator {}.'.format(name), SkipTestWarning)
        return
    for check in _yield_checks(estimator):
        yield check
    if is_classifier(estimator):
        for check in _yield_classifier_checks(estimator):
            yield check
    if is_regressor(estimator):
        for check in _yield_regressor_checks(estimator):
            yield check
    if hasattr(estimator, 'transform'):
        for check in _yield_transformer_checks(estimator):
            yield check
    if isinstance(estimator, ClusterMixin):
        for check in _yield_clustering_checks(estimator):
            yield check
    if is_outlier_detector(estimator):
        for check in _yield_outliers_checks(estimator):
            yield check
    yield check_parameters_default_constructible
    if not tags['non_deterministic']:
        yield check_methods_sample_order_invariance
        yield check_methods_subset_invariance
    yield check_fit2d_1sample
    yield check_fit2d_1feature
    yield check_get_params_invariance
    yield check_set_params
    yield check_dict_unchanged
    yield check_dont_overwrite_parameters
    yield check_fit_idempotent
    yield check_fit_check_is_fitted
    if not tags['no_validation']:
        yield check_n_features_in
        yield check_fit1d
        yield check_fit2d_predict1d
        if tags['requires_y']:
            yield check_requires_y_none
    if tags['requires_positive_X']:
        yield check_fit_non_negative
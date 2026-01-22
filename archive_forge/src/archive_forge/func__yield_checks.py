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
def _yield_checks(estimator):
    name = estimator.__class__.__name__
    tags = _safe_tags(estimator)
    yield check_no_attributes_set_in_init
    yield check_estimators_dtypes
    yield check_fit_score_takes_y
    if has_fit_parameter(estimator, 'sample_weight'):
        yield check_sample_weights_pandas_series
        yield check_sample_weights_not_an_array
        yield check_sample_weights_list
        if not tags['pairwise']:
            yield check_sample_weights_shape
            yield check_sample_weights_not_overwritten
            yield partial(check_sample_weights_invariance, kind='ones')
            yield partial(check_sample_weights_invariance, kind='zeros')
    yield check_estimators_fit_returns_self
    yield partial(check_estimators_fit_returns_self, readonly_memmap=True)
    if not tags['no_validation']:
        yield check_complex_data
        yield check_dtype_object
        yield check_estimators_empty_data_messages
    if name not in CROSS_DECOMPOSITION:
        yield check_pipeline_consistency
    if not tags['allow_nan'] and (not tags['no_validation']):
        yield check_estimators_nan_inf
    if tags['pairwise']:
        yield check_nonsquare_error
    yield check_estimators_overwrite_params
    if hasattr(estimator, 'sparsify'):
        yield check_sparsify_coefficients
    yield check_estimator_sparse_data
    yield check_estimators_pickle
    yield partial(check_estimators_pickle, readonly_memmap=True)
    yield check_estimator_get_tags_default_keys
    if tags['array_api_support']:
        for check in _yield_array_api_checks(estimator):
            yield check
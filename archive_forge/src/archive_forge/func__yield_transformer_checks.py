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
def _yield_transformer_checks(transformer):
    tags = _safe_tags(transformer)
    if not tags['no_validation']:
        yield check_transformer_data_not_an_array
    yield check_transformer_general
    if tags['preserves_dtype']:
        yield check_transformer_preserve_dtypes
    yield partial(check_transformer_general, readonly_memmap=True)
    if not _safe_tags(transformer, key='stateless'):
        yield check_transformers_unfitted
    else:
        yield check_transformers_unfitted_stateless
    external_solver = ['Isomap', 'KernelPCA', 'LocallyLinearEmbedding', 'RandomizedLasso', 'LogisticRegressionCV', 'BisectingKMeans']
    name = transformer.__class__.__name__
    if name not in external_solver:
        yield check_transformer_n_iter
import numbers
import warnings
from collections import Counter
from functools import partial
import numpy as np
import numpy.ma as ma
from scipy import sparse as sp
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import _is_pandas_na, is_scalar_nan
from ..utils._mask import _get_mask
from ..utils._param_validation import MissingValues, StrOptions
from ..utils.fixes import _mode
from ..utils.sparsefuncs import _get_median
from ..utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted
def _check_inputs_dtype(X, missing_values):
    if _is_pandas_na(missing_values):
        return
    if X.dtype.kind in ('f', 'i', 'u') and (not isinstance(missing_values, numbers.Real)):
        raise ValueError("'X' and 'missing_values' types are expected to be both numerical. Got X.dtype={} and  type(missing_values)={}.".format(X.dtype, type(missing_values)))
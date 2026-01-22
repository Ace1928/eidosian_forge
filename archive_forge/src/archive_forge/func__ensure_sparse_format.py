import numbers
import operator
import sys
import warnings
from contextlib import suppress
from functools import reduce, wraps
from inspect import Parameter, isclass, signature
import joblib
import numpy as np
import scipy.sparse as sp
from .. import get_config as _get_config
from ..exceptions import DataConversionWarning, NotFittedError, PositiveSpectrumWarning
from ..utils._array_api import _asarray_with_order, _is_numpy_namespace, get_namespace
from ..utils.fixes import ComplexWarning, _preserve_dia_indices_dtype
from ._isfinite import FiniteStatus, cy_isfinite
from .fixes import _object_dtype_isnan
def _ensure_sparse_format(sparse_container, accept_sparse, dtype, copy, force_all_finite, accept_large_sparse, estimator_name=None, input_name=''):
    """Convert a sparse container to a given format.

    Checks the sparse format of `sparse_container` and converts if necessary.

    Parameters
    ----------
    sparse_container : sparse matrix or array
        Input to validate and convert.

    accept_sparse : str, bool or list/tuple of str
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.

    dtype : str, type or None
        Data type of result. If None, the dtype of the input is preserved.

    copy : bool
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : bool or 'allow-nan'
        Whether to raise an error on np.inf, np.nan, pd.NA in X. The
        possibilities are:

        - True: Force all values of X to be finite.
        - False: accepts np.inf, np.nan, pd.NA in X.
        - 'allow-nan': accepts only np.nan and pd.NA values in X. Values cannot
          be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`


    estimator_name : str, default=None
        The estimator name, used to construct the error message.

    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.

    Returns
    -------
    sparse_container_converted : sparse matrix or array
        Sparse container (matrix/array) that is ensured to have an allowed type.
    """
    if dtype is None:
        dtype = sparse_container.dtype
    changed_format = False
    sparse_container_type_name = type(sparse_container).__name__
    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]
    _check_large_sparse(sparse_container, accept_large_sparse)
    if accept_sparse is False:
        padded_input = ' for ' + input_name if input_name else ''
        raise TypeError(f"Sparse data was passed{padded_input}, but dense data is required. Use '.toarray()' to convert to a dense numpy array.")
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError("When providing 'accept_sparse' as a tuple or list, it must contain at least one string value.")
        if sparse_container.format not in accept_sparse:
            sparse_container = sparse_container.asformat(accept_sparse[0])
            changed_format = True
    elif accept_sparse is not True:
        raise ValueError(f"Parameter 'accept_sparse' should be a string, boolean or list of strings. You provided 'accept_sparse={accept_sparse}'.")
    if dtype != sparse_container.dtype:
        sparse_container = sparse_container.astype(dtype)
    elif copy and (not changed_format):
        sparse_container = sparse_container.copy()
    if force_all_finite:
        if not hasattr(sparse_container, 'data'):
            warnings.warn(f"Can't check {sparse_container.format} sparse matrix for nan or inf.", stacklevel=2)
        else:
            _assert_all_finite(sparse_container.data, allow_nan=force_all_finite == 'allow-nan', estimator_name=estimator_name, input_name=input_name)
    if changed_format:
        requested_sparse_format = accept_sparse[0]
        _preserve_dia_indices_dtype(sparse_container, sparse_container_type_name, requested_sparse_format)
    return sparse_container
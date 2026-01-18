import warnings
from collections.abc import Sequence
from itertools import chain
import numpy as np
from scipy.sparse import issparse
from ..utils._array_api import get_namespace
from ..utils.fixes import VisibleDeprecationWarning
from .validation import _assert_all_finite, check_array
def type_of_target(y, input_name=''):
    """Determine the type of data indicated by the target.

    Note that this type is the most specific type that can be inferred.
    For example:

        * ``binary`` is more specific but compatible with ``multiclass``.
        * ``multiclass`` of integers is more specific but compatible with
          ``continuous``.
        * ``multilabel-indicator`` is more specific but compatible with
          ``multiclass-multioutput``.

    Parameters
    ----------
    y : {array-like, sparse matrix}
        Target values. If a sparse matrix, `y` is expected to be a
        CSR/CSC matrix.

    input_name : str, default=""
        The data name used to construct the error message.

        .. versionadded:: 1.1.0

    Returns
    -------
    target_type : str
        One of:

        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.

    Examples
    --------
    >>> from sklearn.utils.multiclass import type_of_target
    >>> import numpy as np
    >>> type_of_target([0.1, 0.6])
    'continuous'
    >>> type_of_target([1, -1, -1, 1])
    'binary'
    >>> type_of_target(['a', 'b', 'a'])
    'binary'
    >>> type_of_target([1.0, 2.0])
    'binary'
    >>> type_of_target([1, 0, 2])
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0])
    'multiclass'
    >>> type_of_target(['a', 'b', 'c'])
    'multiclass'
    >>> type_of_target(np.array([[1, 2], [3, 1]]))
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]])
    'multilabel-indicator'
    >>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
    'continuous-multioutput'
    >>> type_of_target(np.array([[0, 1], [1, 1]]))
    'multilabel-indicator'
    """
    xp, is_array_api_compliant = get_namespace(y)
    valid = (isinstance(y, Sequence) or issparse(y) or hasattr(y, '__array__')) and (not isinstance(y, str)) or is_array_api_compliant
    if not valid:
        raise ValueError('Expected array-like (array or non-string sequence), got %r' % y)
    sparse_pandas = y.__class__.__name__ in ['SparseSeries', 'SparseArray']
    if sparse_pandas:
        raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")
    if is_multilabel(y):
        return 'multilabel-indicator'
    check_y_kwargs = dict(accept_sparse=True, allow_nd=True, force_all_finite=False, ensure_2d=False, ensure_min_samples=0, ensure_min_features=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', VisibleDeprecationWarning)
        if not issparse(y):
            try:
                y = check_array(y, dtype=None, **check_y_kwargs)
            except (VisibleDeprecationWarning, ValueError) as e:
                if str(e).startswith('Complex data not supported'):
                    raise
                y = check_array(y, dtype=object, **check_y_kwargs)
    try:
        first_row = y[[0], :] if issparse(y) else y[0]
        if not hasattr(first_row, '__array__') and isinstance(first_row, Sequence) and (not isinstance(first_row, str)):
            raise ValueError('You appear to be using a legacy multi-label data representation. Sequence of sequences are no longer supported; use a binary array or sparse matrix instead - the MultiLabelBinarizer transformer can convert to this format.')
    except IndexError:
        pass
    if y.ndim not in (1, 2):
        return 'unknown'
    if not min(y.shape):
        if y.ndim == 1:
            return 'binary'
        return 'unknown'
    if not issparse(y) and y.dtype == object and (not isinstance(y.flat[0], str)):
        return 'unknown'
    if y.ndim == 2 and y.shape[1] > 1:
        suffix = '-multioutput'
    else:
        suffix = ''
    if xp.isdtype(y.dtype, 'real floating'):
        data = y.data if issparse(y) else y
        if xp.any(data != xp.astype(data, int)):
            _assert_all_finite(data, input_name=input_name)
            return 'continuous' + suffix
    if issparse(first_row):
        first_row = first_row.data
    if xp.unique_values(y).shape[0] > 2 or (y.ndim == 2 and len(first_row) > 1):
        return 'multiclass' + suffix
    else:
        return 'binary'
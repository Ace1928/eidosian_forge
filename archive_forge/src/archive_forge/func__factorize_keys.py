from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from typing import (
import uuid
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
from pandas.errors import MergeError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.frame import _merge_doc
from pandas.core.indexes.api import default_index
from pandas.core.sorting import (
def _factorize_keys(lk: ArrayLike, rk: ArrayLike, sort: bool=True) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
    """
    Encode left and right keys as enumerated types.

    This is used to get the join indexers to be used when merging DataFrames.

    Parameters
    ----------
    lk : ndarray, ExtensionArray
        Left key.
    rk : ndarray, ExtensionArray
        Right key.
    sort : bool, defaults to True
        If True, the encoding is done such that the unique elements in the
        keys are sorted.

    Returns
    -------
    np.ndarray[np.intp]
        Left (resp. right if called with `key='right'`) labels, as enumerated type.
    np.ndarray[np.intp]
        Right (resp. left if called with `key='right'`) labels, as enumerated type.
    int
        Number of unique elements in union of left and right labels.

    See Also
    --------
    merge : Merge DataFrame or named Series objects
        with a database-style join.
    algorithms.factorize : Encode the object as an enumerated type
        or categorical variable.

    Examples
    --------
    >>> lk = np.array(["a", "c", "b"])
    >>> rk = np.array(["a", "c"])

    Here, the unique values are `'a', 'b', 'c'`. With the default
    `sort=True`, the encoding will be `{0: 'a', 1: 'b', 2: 'c'}`:

    >>> pd.core.reshape.merge._factorize_keys(lk, rk)
    (array([0, 2, 1]), array([0, 2]), 3)

    With the `sort=False`, the encoding will correspond to the order
    in which the unique elements first appear: `{0: 'a', 1: 'c', 2: 'b'}`:

    >>> pd.core.reshape.merge._factorize_keys(lk, rk, sort=False)
    (array([0, 1, 2]), array([0, 1]), 3)
    """
    if isinstance(lk.dtype, DatetimeTZDtype) and isinstance(rk.dtype, DatetimeTZDtype) or (lib.is_np_dtype(lk.dtype, 'M') and lib.is_np_dtype(rk.dtype, 'M')):
        lk, rk = cast('DatetimeArray', lk)._ensure_matching_resos(rk)
        lk = cast('DatetimeArray', lk)._ndarray
        rk = cast('DatetimeArray', rk)._ndarray
    elif isinstance(lk.dtype, CategoricalDtype) and isinstance(rk.dtype, CategoricalDtype) and (lk.dtype == rk.dtype):
        assert isinstance(lk, Categorical)
        assert isinstance(rk, Categorical)
        rk = lk._encode_with_my_categories(rk)
        lk = ensure_int64(lk.codes)
        rk = ensure_int64(rk.codes)
    elif isinstance(lk, ExtensionArray) and lk.dtype == rk.dtype:
        if isinstance(lk.dtype, ArrowDtype) and is_string_dtype(lk.dtype) or (isinstance(lk.dtype, StringDtype) and lk.dtype.storage in ['pyarrow', 'pyarrow_numpy']):
            import pyarrow as pa
            import pyarrow.compute as pc
            len_lk = len(lk)
            lk = lk._pa_array
            rk = rk._pa_array
            dc = pa.chunked_array(lk.chunks + rk.chunks).combine_chunks().dictionary_encode()
            llab, rlab, count = (pc.fill_null(dc.indices[slice(len_lk)], -1).to_numpy().astype(np.intp, copy=False), pc.fill_null(dc.indices[slice(len_lk, None)], -1).to_numpy().astype(np.intp, copy=False), len(dc.dictionary))
            if sort:
                uniques = dc.dictionary.to_numpy(zero_copy_only=False)
                llab, rlab = _sort_labels(uniques, llab, rlab)
            if dc.null_count > 0:
                lmask = llab == -1
                lany = lmask.any()
                rmask = rlab == -1
                rany = rmask.any()
                if lany:
                    np.putmask(llab, lmask, count)
                if rany:
                    np.putmask(rlab, rmask, count)
                count += 1
            return (llab, rlab, count)
        if not isinstance(lk, BaseMaskedArray) and (not (isinstance(lk.dtype, ArrowDtype) and (is_numeric_dtype(lk.dtype.numpy_dtype) or (is_string_dtype(lk.dtype) and (not sort))))):
            lk, _ = lk._values_for_factorize()
            rk, _ = rk._values_for_factorize()
    if needs_i8_conversion(lk.dtype) and lk.dtype == rk.dtype:
        lk = np.asarray(lk, dtype=np.int64)
        rk = np.asarray(rk, dtype=np.int64)
    klass, lk, rk = _convert_arrays_and_get_rizer_klass(lk, rk)
    rizer = klass(max(len(lk), len(rk)))
    if isinstance(lk, BaseMaskedArray):
        assert isinstance(rk, BaseMaskedArray)
        llab = rizer.factorize(lk._data, mask=lk._mask)
        rlab = rizer.factorize(rk._data, mask=rk._mask)
    elif isinstance(lk, ArrowExtensionArray):
        assert isinstance(rk, ArrowExtensionArray)
        llab = rizer.factorize(lk.to_numpy(na_value=1, dtype=lk.dtype.numpy_dtype), mask=lk.isna())
        rlab = rizer.factorize(rk.to_numpy(na_value=1, dtype=lk.dtype.numpy_dtype), mask=rk.isna())
    else:
        llab = rizer.factorize(lk)
        rlab = rizer.factorize(rk)
    assert llab.dtype == np.dtype(np.intp), llab.dtype
    assert rlab.dtype == np.dtype(np.intp), rlab.dtype
    count = rizer.get_count()
    if sort:
        uniques = rizer.uniques.to_array()
        llab, rlab = _sort_labels(uniques, llab, rlab)
    lmask = llab == -1
    lany = lmask.any()
    rmask = rlab == -1
    rany = rmask.any()
    if lany or rany:
        if lany:
            np.putmask(llab, lmask, count)
        if rany:
            np.putmask(rlab, rmask, count)
        count += 1
    return (llab, rlab, count)
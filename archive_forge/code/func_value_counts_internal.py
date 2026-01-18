from __future__ import annotations
import decimal
import operator
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core.array_algos.take import take_nd
from pandas.core.construction import (
from pandas.core.indexers import validate_indices
def value_counts_internal(values, sort: bool=True, ascending: bool=False, normalize: bool=False, bins=None, dropna: bool=True) -> Series:
    from pandas import Index, Series
    index_name = getattr(values, 'name', None)
    name = 'proportion' if normalize else 'count'
    if bins is not None:
        from pandas.core.reshape.tile import cut
        if isinstance(values, Series):
            values = values._values
        try:
            ii = cut(values, bins, include_lowest=True)
        except TypeError as err:
            raise TypeError('bins argument only works with numeric data.') from err
        result = ii.value_counts(dropna=dropna)
        result.name = name
        result = result[result.index.notna()]
        result.index = result.index.astype('interval')
        result = result.sort_index()
        if dropna and (result._values == 0).all():
            result = result.iloc[0:0]
        counts = np.array([len(ii)])
    elif is_extension_array_dtype(values):
        result = Series(values, copy=False)._values.value_counts(dropna=dropna)
        result.name = name
        result.index.name = index_name
        counts = result._values
        if not isinstance(counts, np.ndarray):
            counts = np.asarray(counts)
    elif isinstance(values, ABCMultiIndex):
        levels = list(range(values.nlevels))
        result = Series(index=values, name=name).groupby(level=levels, dropna=dropna).size()
        result.index.names = values.names
        counts = result._values
    else:
        values = _ensure_arraylike(values, func_name='value_counts')
        keys, counts, _ = value_counts_arraylike(values, dropna)
        if keys.dtype == np.float16:
            keys = keys.astype(np.float32)
        idx = Index(keys)
        if idx.dtype == bool and keys.dtype == object:
            idx = idx.astype(object)
        elif idx.dtype != keys.dtype and idx.dtype != 'string[pyarrow_numpy]':
            warnings.warn('The behavior of value_counts with object-dtype is deprecated. In a future version, this will *not* perform dtype inference on the resulting index. To retain the old behavior, use `result.index = result.index.infer_objects()`', FutureWarning, stacklevel=find_stack_level())
        idx.name = index_name
        result = Series(counts, index=idx, name=name, copy=False)
    if sort:
        result = result.sort_values(ascending=ascending)
    if normalize:
        result = result / counts.sum()
    return result
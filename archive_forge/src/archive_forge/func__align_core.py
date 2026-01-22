from __future__ import annotations
from functools import (
from typing import (
import warnings
import numpy as np
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.generic import (
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation.common import result_type_many
@_filter_special_cases
def _align_core(terms):
    term_index = [i for i, term in enumerate(terms) if hasattr(term.value, 'axes')]
    term_dims = [terms[i].value.ndim for i in term_index]
    from pandas import Series
    ndims = Series(dict(zip(term_index, term_dims)))
    biggest = terms[ndims.idxmax()].value
    typ = biggest._constructor
    axes = biggest.axes
    naxes = len(axes)
    gt_than_one_axis = naxes > 1
    for value in (terms[i].value for i in term_index):
        is_series = isinstance(value, ABCSeries)
        is_series_and_gt_one_axis = is_series and gt_than_one_axis
        for axis, items in enumerate(value.axes):
            if is_series_and_gt_one_axis:
                ax, itm = (naxes - 1, value.index)
            else:
                ax, itm = (axis, items)
            if not axes[ax].is_(itm):
                axes[ax] = axes[ax].union(itm)
    for i, ndim in ndims.items():
        for axis, items in zip(range(ndim), axes):
            ti = terms[i].value
            if hasattr(ti, 'reindex'):
                transpose = isinstance(ti, ABCSeries) and naxes > 1
                reindexer = axes[naxes - 1] if transpose else items
                term_axis_size = len(ti.axes[axis])
                reindexer_size = len(reindexer)
                ordm = np.log10(max(1, abs(reindexer_size - term_axis_size)))
                if ordm >= 1 and reindexer_size >= 10000:
                    w = f'Alignment difference on axis {axis} is larger than an order of magnitude on term {repr(terms[i].name)}, by more than {ordm:.4g}; performance may suffer.'
                    warnings.warn(w, category=PerformanceWarning, stacklevel=find_stack_level())
                obj = ti.reindex(reindexer, axis=axis, copy=False)
                terms[i].update(obj)
        terms[i].update(terms[i].value.values)
    return (typ, _zip_axes_from_type(typ, axes))
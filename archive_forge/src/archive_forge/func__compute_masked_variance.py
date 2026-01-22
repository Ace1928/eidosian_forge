from inspect import signature
from math import prod
import numpy
import pandas
from pandas.api.types import is_scalar
from pandas.core.dtypes.common import is_bool_dtype, is_list_like, is_numeric_dtype
import modin.pandas as pd
from modin.core.dataframe.algebra import Binary, Map, Reduce
from modin.error_message import ErrorMessage
from .utils import try_convert_from_interoperable_type
def _compute_masked_variance(self, mask, output_dtype, axis, ddof):
    if axis == 0 and self._ndim != 1:
        raise NotImplementedError('Masked variance on 2D arrays along axis = 0 is currently unsupported.')
    axis_mean = self.mean(axis, output_dtype, keepdims=True, where=mask)
    target = mask.where(self, numpy.nan)
    if self._ndim == 1:
        axis_mean = axis_mean._to_numpy()[0]
        target = target._query_compiler.sub(axis_mean).pow(2).sum(axis=axis)
    else:
        target = (target - axis_mean)._query_compiler.pow(2).sum(axis=axis)
    num_elems = mask.where(self, 0)._query_compiler.notna().sum(axis=axis, skipna=False)
    num_elems = num_elems.sub(ddof)
    target = target.truediv(num_elems)
    na_propagation_mask = mask.where(self, 0)._query_compiler.sum(axis=axis, skipna=False)
    target = target.where(na_propagation_mask.notna(), numpy.nan)
    return target
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
def _less(self, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    check_kwargs(where=where, casting=casting, order=order, subok=subok)
    if is_scalar(x2):
        return array(_query_compiler=self._query_compiler.lt(x2), _ndim=self._ndim)
    caller, callee, new_ndim, kwargs = self._preprocess_binary_op(x2, cast_input_types=False, dtype=dtype, out=out)
    if caller != self._query_compiler:
        result = caller.gt(callee, **kwargs)
    else:
        result = caller.lt(callee, **kwargs)
    return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)
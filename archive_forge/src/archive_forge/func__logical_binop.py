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
def _logical_binop(self, qc_method_name, x2, out, where, casting, order, dtype, subok):
    check_kwargs(where=where, casting=casting, order=order, subok=subok)
    if self._ndim != x2._ndim:
        raise ValueError('modin.numpy logic operators do not currently support broadcasting between arrays of different dimensions')
    caller, callee, new_ndim, kwargs = self._preprocess_binary_op(x2, cast_input_types=False, dtype=dtype, out=out)
    result = getattr(caller, qc_method_name)(callee)
    return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)
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
def align_terms(terms):
    """
    Align a set of terms.
    """
    try:
        terms = list(com.flatten(terms))
    except TypeError:
        if isinstance(terms.value, (ABCSeries, ABCDataFrame)):
            typ = type(terms.value)
            return (typ, _zip_axes_from_type(typ, terms.value.axes))
        return (np.result_type(terms.type), None)
    if all((term.is_scalar for term in terms)):
        return (result_type_many(*(term.value for term in terms)).type, None)
    typ, axes = _align_core(terms)
    return (typ, axes)
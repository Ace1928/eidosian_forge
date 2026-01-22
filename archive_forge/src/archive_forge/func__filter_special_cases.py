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
def _filter_special_cases(f) -> Callable[[F], F]:

    @wraps(f)
    def wrapper(terms):
        if len(terms) == 1:
            return _align_core_single_unary_op(terms[0])
        term_values = (term.value for term in terms)
        if not _any_pandas_objects(terms):
            return (result_type_many(*term_values), None)
        return f(terms)
    return wrapper
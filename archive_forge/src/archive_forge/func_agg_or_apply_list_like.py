from __future__ import annotations
import abc
from collections import defaultdict
import functools
from functools import partial
import inspect
from typing import (
import warnings
import numpy as np
from pandas._config import option_context
from pandas._libs import lib
from pandas._libs.internals import BlockValuesRefs
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SpecificationError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import is_nested_object
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core._numba.executor import generate_apply_looper
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike
def agg_or_apply_list_like(self, op_name: Literal['agg', 'apply']) -> DataFrame | Series:
    obj = self.obj
    kwargs = self.kwargs
    if op_name == 'apply':
        kwargs = {**kwargs, 'by_row': False}
    if getattr(obj, 'axis', 0) == 1:
        raise NotImplementedError('axis other than 0 is not supported')
    if obj._selected_obj.ndim == 1:
        selected_obj = obj._selected_obj
    else:
        selected_obj = obj._obj_with_exclusions
    with com.temp_setattr(obj, 'as_index', True, condition=hasattr(obj, 'as_index')):
        keys, results = self.compute_list_like(op_name, selected_obj, kwargs)
    result = self.wrap_results_list_like(keys, results)
    return result
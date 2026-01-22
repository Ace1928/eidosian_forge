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
class NDFrameApply(Apply):
    """
    Methods shared by FrameApply and SeriesApply but
    not GroupByApply or ResamplerWindowApply
    """
    obj: DataFrame | Series

    @property
    def index(self) -> Index:
        return self.obj.index

    @property
    def agg_axis(self) -> Index:
        return self.obj._get_agg_axis(self.axis)

    def agg_or_apply_list_like(self, op_name: Literal['agg', 'apply']) -> DataFrame | Series:
        obj = self.obj
        kwargs = self.kwargs
        if op_name == 'apply':
            if isinstance(self, FrameApply):
                by_row = self.by_row
            elif isinstance(self, SeriesApply):
                by_row = '_compat' if self.by_row else False
            else:
                by_row = False
            kwargs = {**kwargs, 'by_row': by_row}
        if getattr(obj, 'axis', 0) == 1:
            raise NotImplementedError('axis other than 0 is not supported')
        keys, results = self.compute_list_like(op_name, obj, kwargs)
        result = self.wrap_results_list_like(keys, results)
        return result

    def agg_or_apply_dict_like(self, op_name: Literal['agg', 'apply']) -> DataFrame | Series:
        assert op_name in ['agg', 'apply']
        obj = self.obj
        kwargs = {}
        if op_name == 'apply':
            by_row = '_compat' if self.by_row else False
            kwargs.update({'by_row': by_row})
        if getattr(obj, 'axis', 0) == 1:
            raise NotImplementedError('axis other than 0 is not supported')
        selection = None
        result_index, result_data = self.compute_dict_like(op_name, obj, selection, kwargs)
        result = self.wrap_results_dict_like(obj, result_index, result_data)
        return result
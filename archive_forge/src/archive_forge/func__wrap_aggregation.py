from __future__ import annotations
import warnings
from collections.abc import Iterable
from types import BuiltinFunctionType
from typing import TYPE_CHECKING, Union
import numpy as np
import pandas
import pandas.core.common as com
import pandas.core.groupby
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.dtypes.common import (
from pandas.errors import SpecificationError
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, disable_logging
from modin.pandas.utils import cast_function_modin2pandas
from modin.utils import (
from .series import Series
from .utils import is_label
from .window import RollingGroupby
def _wrap_aggregation(self, qc_method, numeric_only=False, agg_args=None, agg_kwargs=None, **kwargs):
    """
        Perform common metadata transformations and apply groupby functions.

        Parameters
        ----------
        qc_method : callable
            The query compiler method to call.
        numeric_only : {None, True, False}, default: None
            Specifies whether to aggregate non numeric columns:
                - True: include only numeric columns (including categories that holds a numeric dtype)
                - False: include all columns
                - None: infer the parameter, ``False`` if there are no numeric types in the frame,
                  ``True`` otherwise.
        agg_args : list-like, optional
            Positional arguments to pass to the aggregation function.
        agg_kwargs : dict-like, optional
            Keyword arguments to pass to the aggregation function.
        **kwargs : dict
            Keyword arguments to pass to the specified query compiler's method.

        Returns
        -------
        DataFrame or Series
            Returns the same type as `self._df`.
        """
    agg_args = tuple() if agg_args is None else agg_args
    agg_kwargs = dict() if agg_kwargs is None else agg_kwargs
    if numeric_only and self.ndim == 2:
        by_cols = self._internal_by
        mask_cols = [col for col, dtype in self._query_compiler.dtypes.items() if is_numeric_dtype(dtype) or col in by_cols]
        groupby_qc = self._query_compiler.getitem_column_array(mask_cols)
    else:
        groupby_qc = self._query_compiler
    return type(self._df)(query_compiler=qc_method(groupby_qc, by=self._by, axis=self._axis, groupby_kwargs=self._kwargs, agg_args=agg_args, agg_kwargs=agg_kwargs, drop=self._drop, **kwargs))
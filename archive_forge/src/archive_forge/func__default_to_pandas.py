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
def _default_to_pandas(self, f, *args, **kwargs):
    """
        Execute function `f` in default-to-pandas way.

        Parameters
        ----------
        f : callable or str
            The function to apply to each group.
        *args : list
            Extra positional arguments to pass to `f`.
        **kwargs : dict
            Extra keyword arguments to pass to `f`.

        Returns
        -------
        modin.pandas.DataFrame
            A new Modin DataFrame with the result of the pandas function.
        """
    if isinstance(self._by, type(self._query_compiler)) and len(self._by.columns) == 1:
        by = self._by.columns[0] if self._drop else self._by.to_pandas().squeeze()
    elif self._drop and isinstance(self._by, type(self._query_compiler)):
        by = list(self._by.columns)
    else:
        by = self._by
    by = try_cast_to_pandas(by, squeeze=True)
    by = GroupBy.validate_by(by)

    def groupby_on_multiple_columns(df, *args, **kwargs):
        groupby_obj = df.groupby(by=by, axis=self._axis, **self._kwargs)
        if callable(f):
            return f(groupby_obj, *args, **kwargs)
        else:
            ErrorMessage.catch_bugs_and_request_email(failure_condition=not isinstance(f, str))
            attribute = getattr(groupby_obj, f)
            if callable(attribute):
                return attribute(*args, **kwargs)
            return attribute
    return self._df._default_to_pandas(groupby_on_multiple_columns, *args, **kwargs)
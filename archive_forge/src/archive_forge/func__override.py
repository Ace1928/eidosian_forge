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
def _override(self, **kwargs):
    """
        Override groupby parameters.

        Parameters
        ----------
        **kwargs : dict
            Parameters to override.

        Returns
        -------
        DataFrameGroupBy
            A groupby object with new parameters.
        """
    new_kw = dict(df=self._df, by=self._by, axis=self._axis, idx_name=self._idx_name, drop=self._drop, **self._kwargs)
    new_kw.update(kwargs)
    return type(self)(**new_kw)
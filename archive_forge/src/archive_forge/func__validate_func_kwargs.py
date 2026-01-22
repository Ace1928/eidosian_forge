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
def _validate_func_kwargs(self, kwargs: dict):
    """
        Validate types of user-provided "named aggregation" kwargs.

        Parameters
        ----------
        kwargs : dict

        Returns
        -------
        columns : List[str]
            List of user-provided keys.
        funcs : List[Union[str, callable[...,Any]]]
            List of user-provided aggfuncs.

        Raises
        ------
        `TypeError` is raised if aggfunc is not `str` or callable.

        Notes
        -----
        Copied from pandas.
        """
    columns = list(kwargs)
    funcs = []
    for col_func in kwargs.values():
        if not (isinstance(col_func, str) or callable(col_func)):
            raise TypeError(f'func is expected but received {type(col_func).__name__} in **kwargs.')
        funcs.append(col_func)
    if not columns:
        raise TypeError("Must provide 'func' or named aggregation **kwargs.")
    return (columns, funcs)
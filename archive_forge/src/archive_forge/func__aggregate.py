from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
import pandas.core.window.rolling
from pandas.core.dtypes.common import is_list_like
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.utils import cast_function_modin2pandas
from modin.utils import _inherit_docstrings
def _aggregate(self, method_name, *args, as_index=None, **kwargs):
    """
        Run the specified rolling aggregation.

        Parameters
        ----------
        method_name : str
            Name of the aggregation.
        *args : tuple
            Positional arguments to pass to the aggregation.
        as_index : bool, optional
            Whether the result should have the group labels as index levels or as columns.
            If not specified the parameter value will be taken from groupby kwargs.
        **kwargs : dict
            Keyword arguments to pass to the aggregation.

        Returns
        -------
        DataFrame or Series
            Result of the aggregation.
        """
    res = self._groupby_obj._wrap_aggregation(qc_method=type(self._query_compiler).groupby_rolling, numeric_only=False, agg_args=args, agg_kwargs=kwargs, agg_func=method_name, rolling_kwargs=self.rolling_kwargs)
    if as_index is None:
        as_index = self._as_index
    if not as_index:
        res = res.reset_index(level=[i for i in range(len(self._groupby_obj._internal_by))], drop=False)
    return res
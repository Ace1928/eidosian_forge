from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
def _update_proxy(self, parent, column_name):
    """
        Create a new proxy, if either parent or column name are different.

        Parameters
        ----------
        parent : object
            Source object to extract categories on demand.
        column_name : str
            Column name of the categorical column in the source object.

        Returns
        -------
        pandas.CategoricalDtype or LazyProxyCategoricalDtype
        """
    if self._is_materialized:
        return pandas.CategoricalDtype(self.categories, ordered=self._ordered)
    elif parent is self._parent and column_name == self._column_name:
        return self
    else:
        return self._build_proxy(parent, column_name, self._materializer)
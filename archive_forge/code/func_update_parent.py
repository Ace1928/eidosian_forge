from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
def update_parent(self, new_parent: 'PandasDataframe'):
    """
        Set new parent dataframe.

        Parameters
        ----------
        new_parent : PandasDataframe
        """
    self._parent_df = new_parent
    LazyProxyCategoricalDtype.update_dtypes(self._known_dtypes, new_parent)
    self.columns_order
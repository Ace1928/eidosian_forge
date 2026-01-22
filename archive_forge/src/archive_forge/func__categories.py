from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
@_categories.setter
def _categories(self, categories):
    """
        Set new categorical values.

        Parameters
        ----------
        categories : list-like
        """
    self._categories_val = categories
    self._parent = None
    self._materializer = None
    self._dtype = None
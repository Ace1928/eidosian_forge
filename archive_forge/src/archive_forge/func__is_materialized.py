from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
@property
def _is_materialized(self) -> bool:
    """
        Check whether categorical values were already materialized.

        Returns
        -------
        bool
        """
    return self._categories_val is not None
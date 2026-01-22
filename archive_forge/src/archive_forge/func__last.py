from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _last(self):
    """
        Get the last node of the resulting calcite node sequence.

        Returns
        -------
        CalciteBaseNode
        """
    return self.res[-1]
import collections
from typing import Any, Dict, Iterable, Optional, Sequence
import numpy as np
import pyarrow as pa
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.error_message import ErrorMessage
from modin.experimental.core.execution.native.implementations.hdk_on_native.dataframe.dataframe import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.df_algebra import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .column import HdkProtocolColumn
from .utils import raise_copy_alert_if_materialize
@property
def _is_zero_copy_possible(self) -> bool:
    """
        Check whether it's possible to retrieve data from the DataFrame zero-copy.

        The 'zero-copy' term also means that no extra computations or data transers
        are needed to access the data.

        Returns
        -------
        bool
        """
    if self.__is_zero_copy_possible is None:
        if self._df._has_arrow_table():
            self.__is_zero_copy_possible = True
        elif not self._df._can_execute_arrow():
            self.__is_zero_copy_possible = False
        else:
            self.__is_zero_copy_possible = self._is_zero_copy_arrow_op(self._df._op)
    return self.__is_zero_copy_possible
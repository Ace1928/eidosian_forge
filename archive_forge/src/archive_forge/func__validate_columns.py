from __future__ import annotations
import codecs
import io
from typing import (
import warnings
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.missing import isna
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
from pandas.io.xml import (
@final
def _validate_columns(self) -> None:
    """
        Validate elems_cols and attrs_cols.

        This method will check if columns is list-like.

        Raises
        ------
        ValueError
            * If value is not a list and less then length of nodes.
        """
    if self.attr_cols and (not is_list_like(self.attr_cols)):
        raise TypeError(f'{type(self.attr_cols).__name__} is not a valid type for attr_cols')
    if self.elem_cols and (not is_list_like(self.elem_cols)):
        raise TypeError(f'{type(self.elem_cols).__name__} is not a valid type for elem_cols')
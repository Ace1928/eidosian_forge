from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
def __dataframe__(self, nan_as_null: bool=False, allow_copy: bool=True) -> PandasDataFrameXchg:
    return PandasDataFrameXchg(self._df, allow_copy)
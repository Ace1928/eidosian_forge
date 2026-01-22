from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
class DataFrameInfo(_BaseInfo):
    """
    Class storing dataframe-specific info.
    """

    def __init__(self, data: DataFrame, memory_usage: bool | str | None=None) -> None:
        self.data: DataFrame = data
        self.memory_usage = _initialize_memory_usage(memory_usage)

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        return _get_dataframe_dtype_counts(self.data)

    @property
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes
            Dtype of each of the DataFrame's columns.
        """
        return self.data.dtypes

    @property
    def ids(self) -> Index:
        """
        Column names.

        Returns
        -------
        ids : Index
            DataFrame's column names.
        """
        return self.data.columns

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return len(self.ids)

    @property
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""
        return self.data.count()

    @property
    def memory_usage_bytes(self) -> int:
        deep = self.memory_usage == 'deep'
        return self.data.memory_usage(index=True, deep=deep).sum()

    def render(self, *, buf: WriteBuffer[str] | None, max_cols: int | None, verbose: bool | None, show_counts: bool | None) -> None:
        printer = _DataFrameInfoPrinter(info=self, max_cols=max_cols, verbose=verbose, show_counts=show_counts)
        printer.to_buffer(buf)
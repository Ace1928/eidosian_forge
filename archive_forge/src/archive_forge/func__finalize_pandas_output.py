from __future__ import annotations
from typing import TYPE_CHECKING
import warnings
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.inference import is_integer
import pandas as pd
from pandas import DataFrame
from pandas.io._util import (
from pandas.io.parsers.base_parser import ParserBase
def _finalize_pandas_output(self, frame: DataFrame) -> DataFrame:
    """
        Processes data read in based on kwargs.

        Parameters
        ----------
        frame: DataFrame
            The DataFrame to process.

        Returns
        -------
        DataFrame
            The processed DataFrame.
        """
    num_cols = len(frame.columns)
    multi_index_named = True
    if self.header is None:
        if self.names is None:
            if self.header is None:
                self.names = range(num_cols)
        if len(self.names) != num_cols:
            self.names = list(range(num_cols - len(self.names))) + self.names
            multi_index_named = False
        frame.columns = self.names
    _, frame = self._do_date_conversions(frame.columns, frame)
    if self.index_col is not None:
        index_to_set = self.index_col.copy()
        for i, item in enumerate(self.index_col):
            if is_integer(item):
                index_to_set[i] = frame.columns[item]
            elif item not in frame.columns:
                raise ValueError(f'Index {item} invalid')
            if self.dtype is not None:
                key, new_dtype = (item, self.dtype.get(item)) if self.dtype.get(item) is not None else (frame.columns[item], self.dtype.get(frame.columns[item]))
                if new_dtype is not None:
                    frame[key] = frame[key].astype(new_dtype)
                    del self.dtype[key]
        frame.set_index(index_to_set, drop=True, inplace=True)
        if self.header is None and (not multi_index_named):
            frame.index.names = [None] * len(frame.index.names)
    if self.dtype is not None:
        if isinstance(self.dtype, dict):
            self.dtype = {k: pandas_dtype(v) for k, v in self.dtype.items() if k in frame.columns}
        else:
            self.dtype = pandas_dtype(self.dtype)
        try:
            frame = frame.astype(self.dtype)
        except TypeError as e:
            raise ValueError(e)
    return frame
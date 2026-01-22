from __future__ import annotations
from collections import (
from collections.abc import (
import csv
from io import StringIO
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.common import (
from pandas.io.parsers.base_parser import (
def _get_index_name(self) -> tuple[Sequence[Hashable] | None, list[Hashable], list[Hashable]]:
    """
        Try several cases to get lines:

        0) There are headers on row 0 and row 1 and their
        total summed lengths equals the length of the next line.
        Treat row 0 as columns and row 1 as indices
        1) Look for implicit index: there are more columns
        on row 1 than row 0. If this is true, assume that row
        1 lists index columns and row 0 lists normal columns.
        2) Get index from the columns if it was listed.
        """
    columns: Sequence[Hashable] = self.orig_names
    orig_names = list(columns)
    columns = list(columns)
    line: list[Scalar] | None
    if self._header_line is not None:
        line = self._header_line
    else:
        try:
            line = self._next_line()
        except StopIteration:
            line = None
    next_line: list[Scalar] | None
    try:
        next_line = self._next_line()
    except StopIteration:
        next_line = None
    implicit_first_cols = 0
    if line is not None:
        index_col = self.index_col
        if index_col is not False:
            implicit_first_cols = len(line) - self.num_original_columns
        if next_line is not None and self.header is not None and (index_col is not False):
            if len(next_line) == len(line) + self.num_original_columns:
                self.index_col = list(range(len(line)))
                self.buf = self.buf[1:]
                for c in reversed(line):
                    columns.insert(0, c)
                orig_names = list(columns)
                self.num_original_columns = len(columns)
                return (line, orig_names, columns)
    if implicit_first_cols > 0:
        self._implicit_index = True
        if self.index_col is None:
            self.index_col = list(range(implicit_first_cols))
        index_name = None
    else:
        index_name, _, self.index_col = self._clean_index_names(columns, self.index_col)
    return (index_name, orig_names, columns)
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
class FixedWidthReader(abc.Iterator):
    """
    A reader of fixed-width lines.
    """

    def __init__(self, f: IO[str] | ReadCsvBuffer[str], colspecs: list[tuple[int, int]] | Literal['infer'], delimiter: str | None, comment: str | None, skiprows: set[int] | None=None, infer_nrows: int=100) -> None:
        self.f = f
        self.buffer: Iterator | None = None
        self.delimiter = '\r\n' + delimiter if delimiter else '\n\r\t '
        self.comment = comment
        if colspecs == 'infer':
            self.colspecs = self.detect_colspecs(infer_nrows=infer_nrows, skiprows=skiprows)
        else:
            self.colspecs = colspecs
        if not isinstance(self.colspecs, (tuple, list)):
            raise TypeError(f'column specifications must be a list or tuple, input was a {type(colspecs).__name__}')
        for colspec in self.colspecs:
            if not (isinstance(colspec, (tuple, list)) and len(colspec) == 2 and isinstance(colspec[0], (int, np.integer, type(None))) and isinstance(colspec[1], (int, np.integer, type(None)))):
                raise TypeError('Each column specification must be 2 element tuple or list of integers')

    def get_rows(self, infer_nrows: int, skiprows: set[int] | None=None) -> list[str]:
        """
        Read rows from self.f, skipping as specified.

        We distinguish buffer_rows (the first <= infer_nrows
        lines) from the rows returned to detect_colspecs
        because it's simpler to leave the other locations
        with skiprows logic alone than to modify them to
        deal with the fact we skipped some rows here as
        well.

        Parameters
        ----------
        infer_nrows : int
            Number of rows to read from self.f, not counting
            rows that are skipped.
        skiprows: set, optional
            Indices of rows to skip.

        Returns
        -------
        detect_rows : list of str
            A list containing the rows to read.

        """
        if skiprows is None:
            skiprows = set()
        buffer_rows = []
        detect_rows = []
        for i, row in enumerate(self.f):
            if i not in skiprows:
                detect_rows.append(row)
            buffer_rows.append(row)
            if len(detect_rows) >= infer_nrows:
                break
        self.buffer = iter(buffer_rows)
        return detect_rows

    def detect_colspecs(self, infer_nrows: int=100, skiprows: set[int] | None=None) -> list[tuple[int, int]]:
        delimiters = ''.join([f'\\{x}' for x in self.delimiter])
        pattern = re.compile(f'([^{delimiters}]+)')
        rows = self.get_rows(infer_nrows, skiprows)
        if not rows:
            raise EmptyDataError('No rows from which to infer column width')
        max_len = max(map(len, rows))
        mask = np.zeros(max_len + 1, dtype=int)
        if self.comment is not None:
            rows = [row.partition(self.comment)[0] for row in rows]
        for row in rows:
            for m in pattern.finditer(row):
                mask[m.start():m.end()] = 1
        shifted = np.roll(mask, 1)
        shifted[0] = 0
        edges = np.where(mask ^ shifted == 1)[0]
        edge_pairs = list(zip(edges[::2], edges[1::2]))
        return edge_pairs

    def __next__(self) -> list[str]:
        if self.buffer is not None:
            try:
                line = next(self.buffer)
            except StopIteration:
                self.buffer = None
                line = next(self.f)
        else:
            line = next(self.f)
        return [line[from_:to].strip(self.delimiter) for from_, to in self.colspecs]
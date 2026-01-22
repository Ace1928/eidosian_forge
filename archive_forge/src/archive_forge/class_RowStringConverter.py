from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
class RowStringConverter:
    """Converter for dataframe rows into LaTeX strings.

    Parameters
    ----------
    formatter : `DataFrameFormatter`
        Instance of `DataFrameFormatter`.
    multicolumn: bool, optional
        Whether to use \\multicolumn macro.
    multicolumn_format: str, optional
        Multicolumn format.
    multirow: bool, optional
        Whether to use \\multirow macro.

    """

    def __init__(self, formatter: DataFrameFormatter, multicolumn: bool=False, multicolumn_format: str | None=None, multirow: bool=False) -> None:
        self.fmt = formatter
        self.frame = self.fmt.frame
        self.multicolumn = multicolumn
        self.multicolumn_format = multicolumn_format
        self.multirow = multirow
        self.clinebuf: list[list[int]] = []
        self.strcols = self._get_strcols()
        self.strrows = list(zip(*self.strcols))

    def get_strrow(self, row_num: int) -> str:
        """Get string representation of the row."""
        row = self.strrows[row_num]
        is_multicol = row_num < self.column_levels and self.fmt.header and self.multicolumn
        is_multirow = row_num >= self.header_levels and self.fmt.index and self.multirow and (self.index_levels > 1)
        is_cline_maybe_required = is_multirow and row_num < len(self.strrows) - 1
        crow = self._preprocess_row(row)
        if is_multicol:
            crow = self._format_multicolumn(crow)
        if is_multirow:
            crow = self._format_multirow(crow, row_num)
        lst = []
        lst.append(' & '.join(crow))
        lst.append(' \\\\')
        if is_cline_maybe_required:
            cline = self._compose_cline(row_num, len(self.strcols))
            lst.append(cline)
        return ''.join(lst)

    @property
    def _header_row_num(self) -> int:
        """Number of rows in header."""
        return self.header_levels if self.fmt.header else 0

    @property
    def index_levels(self) -> int:
        """Integer number of levels in index."""
        return self.frame.index.nlevels

    @property
    def column_levels(self) -> int:
        return self.frame.columns.nlevels

    @property
    def header_levels(self) -> int:
        nlevels = self.column_levels
        if self.fmt.has_index_names and self.fmt.show_index_names:
            nlevels += 1
        return nlevels

    def _get_strcols(self) -> list[list[str]]:
        """String representation of the columns."""
        if self.fmt.frame.empty:
            strcols = [[self._empty_info_line]]
        else:
            strcols = self.fmt.get_strcols()
        if self.fmt.index and isinstance(self.frame.index, ABCMultiIndex):
            out = self.frame.index.format(adjoin=False, sparsify=self.fmt.sparsify, names=self.fmt.has_index_names, na_rep=self.fmt.na_rep)

            def pad_empties(x):
                for pad in reversed(x):
                    if pad:
                        return [x[0]] + [i if i else ' ' * len(pad) for i in x[1:]]
            gen = (pad_empties(i) for i in out)
            clevels = self.frame.columns.nlevels
            out = [[' ' * len(i[-1])] * clevels + i for i in gen]
            cnames = self.frame.columns.names
            if any(cnames):
                new_names = [i if i else '{}' for i in cnames]
                out[self.frame.index.nlevels - 1][:clevels] = new_names
            strcols = out + strcols[1:]
        return strcols

    @property
    def _empty_info_line(self) -> str:
        return f'Empty {type(self.frame).__name__}\nColumns: {self.frame.columns}\nIndex: {self.frame.index}'

    def _preprocess_row(self, row: Sequence[str]) -> list[str]:
        """Preprocess elements of the row."""
        if self.fmt.escape:
            crow = _escape_symbols(row)
        else:
            crow = [x if x else '{}' for x in row]
        if self.fmt.bold_rows and self.fmt.index:
            crow = _convert_to_bold(crow, self.index_levels)
        return crow

    def _format_multicolumn(self, row: list[str]) -> list[str]:
        """
        Combine columns belonging to a group to a single multicolumn entry
        according to self.multicolumn_format

        e.g.:
        a &  &  & b & c &
        will become
        \\multicolumn{3}{l}{a} & b & \\multicolumn{2}{l}{c}
        """
        row2 = row[:self.index_levels]
        ncol = 1
        coltext = ''

        def append_col() -> None:
            if ncol > 1:
                row2.append(f'\\multicolumn{{{ncol:d}}}{{{self.multicolumn_format}}}{{{coltext.strip()}}}')
            else:
                row2.append(coltext)
        for c in row[self.index_levels:]:
            if c.strip():
                if coltext:
                    append_col()
                coltext = c
                ncol = 1
            else:
                ncol += 1
        if coltext:
            append_col()
        return row2

    def _format_multirow(self, row: list[str], i: int) -> list[str]:
        """
        Check following rows, whether row should be a multirow

        e.g.:     becomes:
        a & 0 &   \\multirow{2}{*}{a} & 0 &
          & 1 &     & 1 &
        b & 0 &   \\cline{1-2}
                  b & 0 &
        """
        for j in range(self.index_levels):
            if row[j].strip():
                nrow = 1
                for r in self.strrows[i + 1:]:
                    if not r[j].strip():
                        nrow += 1
                    else:
                        break
                if nrow > 1:
                    row[j] = f'\\multirow{{{nrow:d}}}{{*}}{{{row[j].strip()}}}'
                    self.clinebuf.append([i + nrow - 1, j + 1])
        return row

    def _compose_cline(self, i: int, icol: int) -> str:
        """
        Create clines after multirow-blocks are finished.
        """
        lst = []
        for cl in self.clinebuf:
            if cl[0] == i:
                lst.append(f'\n\\cline{{{cl[1]:d}-{icol:d}}}')
                self.clinebuf = [x for x in self.clinebuf if x[0] != i]
        return ''.join(lst)
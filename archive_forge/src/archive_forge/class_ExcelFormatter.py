from __future__ import annotations
from collections.abc import (
import functools
import itertools
import re
from typing import (
import warnings
import numpy as np
from pandas._libs.lib import is_list_like
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes import missing
from pandas.core.dtypes.common import (
from pandas import (
import pandas.core.common as com
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats._color_data import CSS4_COLORS
from pandas.io.formats.css import (
from pandas.io.formats.format import get_level_lengths
from pandas.io.formats.printing import pprint_thing
class ExcelFormatter:
    """
    Class for formatting a DataFrame to a list of ExcelCells,

    Parameters
    ----------
    df : DataFrame or Styler
    na_rep: na representation
    float_format : str, default None
        Format string for floating point numbers
    cols : sequence, optional
        Columns to write
    header : bool or sequence of str, default True
        Write out column names. If a list of string is given it is
        assumed to be aliases for the column names
    index : bool, default True
        output row names (index)
    index_label : str or sequence, default None
        Column label for index column(s) if desired. If None is given, and
        `header` and `index` are True, then the index names are used. A
        sequence should be given if the DataFrame uses MultiIndex.
    merge_cells : bool, default False
        Format MultiIndex and Hierarchical Rows as merged cells.
    inf_rep : str, default `'inf'`
        representation for np.inf values (which aren't representable in Excel)
        A `'-'` sign will be added in front of -inf.
    style_converter : callable, optional
        This translates Styler styles (CSS) into ExcelWriter styles.
        Defaults to ``CSSToExcelConverter()``.
        It should have signature css_declarations string -> excel style.
        This is only called for body cells.
    """
    max_rows = 2 ** 20
    max_cols = 2 ** 14

    def __init__(self, df, na_rep: str='', float_format: str | None=None, cols: Sequence[Hashable] | None=None, header: Sequence[Hashable] | bool=True, index: bool=True, index_label: IndexLabel | None=None, merge_cells: bool=False, inf_rep: str='inf', style_converter: Callable | None=None) -> None:
        self.rowcounter = 0
        self.na_rep = na_rep
        if not isinstance(df, DataFrame):
            self.styler = df
            self.styler._compute()
            df = df.data
            if style_converter is None:
                style_converter = CSSToExcelConverter()
            self.style_converter: Callable | None = style_converter
        else:
            self.styler = None
            self.style_converter = None
        self.df = df
        if cols is not None:
            if not len(Index(cols).intersection(df.columns)):
                raise KeyError('passes columns are not ALL present dataframe')
            if len(Index(cols).intersection(df.columns)) != len(set(cols)):
                raise KeyError("Not all names specified in 'columns' are found")
            self.df = df.reindex(columns=cols)
        self.columns = self.df.columns
        self.float_format = float_format
        self.index = index
        self.index_label = index_label
        self.header = header
        self.merge_cells = merge_cells
        self.inf_rep = inf_rep

    @property
    def header_style(self) -> dict[str, dict[str, str | bool]]:
        return {'font': {'bold': True}, 'borders': {'top': 'thin', 'right': 'thin', 'bottom': 'thin', 'left': 'thin'}, 'alignment': {'horizontal': 'center', 'vertical': 'top'}}

    def _format_value(self, val):
        if is_scalar(val) and missing.isna(val):
            val = self.na_rep
        elif is_float(val):
            if missing.isposinf_scalar(val):
                val = self.inf_rep
            elif missing.isneginf_scalar(val):
                val = f'-{self.inf_rep}'
            elif self.float_format is not None:
                val = float(self.float_format % val)
        if getattr(val, 'tzinfo', None) is not None:
            raise ValueError('Excel does not support datetimes with timezones. Please ensure that datetimes are timezone unaware before writing to Excel.')
        return val

    def _format_header_mi(self) -> Iterable[ExcelCell]:
        if self.columns.nlevels > 1:
            if not self.index:
                raise NotImplementedError("Writing to Excel with MultiIndex columns and no index ('index'=False) is not yet implemented.")
        if not (self._has_aliases or self.header):
            return
        columns = self.columns
        level_strs = columns._format_multi(sparsify=self.merge_cells, include_names=False)
        level_lengths = get_level_lengths(level_strs)
        coloffset = 0
        lnum = 0
        if self.index and isinstance(self.df.index, MultiIndex):
            coloffset = len(self.df.index[0]) - 1
        if self.merge_cells:
            for lnum, name in enumerate(columns.names):
                yield ExcelCell(row=lnum, col=coloffset, val=name, style=self.header_style)
            for lnum, (spans, levels, level_codes) in enumerate(zip(level_lengths, columns.levels, columns.codes)):
                values = levels.take(level_codes)
                for i, span_val in spans.items():
                    mergestart, mergeend = (None, None)
                    if span_val > 1:
                        mergestart, mergeend = (lnum, coloffset + i + span_val)
                    yield CssExcelCell(row=lnum, col=coloffset + i + 1, val=values[i], style=self.header_style, css_styles=getattr(self.styler, 'ctx_columns', None), css_row=lnum, css_col=i, css_converter=self.style_converter, mergestart=mergestart, mergeend=mergeend)
        else:
            for i, values in enumerate(zip(*level_strs)):
                v = '.'.join(map(pprint_thing, values))
                yield CssExcelCell(row=lnum, col=coloffset + i + 1, val=v, style=self.header_style, css_styles=getattr(self.styler, 'ctx_columns', None), css_row=lnum, css_col=i, css_converter=self.style_converter)
        self.rowcounter = lnum

    def _format_header_regular(self) -> Iterable[ExcelCell]:
        if self._has_aliases or self.header:
            coloffset = 0
            if self.index:
                coloffset = 1
                if isinstance(self.df.index, MultiIndex):
                    coloffset = len(self.df.index.names)
            colnames = self.columns
            if self._has_aliases:
                self.header = cast(Sequence, self.header)
                if len(self.header) != len(self.columns):
                    raise ValueError(f'Writing {len(self.columns)} cols but got {len(self.header)} aliases')
                colnames = self.header
            for colindex, colname in enumerate(colnames):
                yield CssExcelCell(row=self.rowcounter, col=colindex + coloffset, val=colname, style=self.header_style, css_styles=getattr(self.styler, 'ctx_columns', None), css_row=0, css_col=colindex, css_converter=self.style_converter)

    def _format_header(self) -> Iterable[ExcelCell]:
        gen: Iterable[ExcelCell]
        if isinstance(self.columns, MultiIndex):
            gen = self._format_header_mi()
        else:
            gen = self._format_header_regular()
        gen2: Iterable[ExcelCell] = ()
        if self.df.index.names:
            row = [x if x is not None else '' for x in self.df.index.names] + [''] * len(self.columns)
            if functools.reduce(lambda x, y: x and y, (x != '' for x in row)):
                gen2 = (ExcelCell(self.rowcounter, colindex, val, self.header_style) for colindex, val in enumerate(row))
                self.rowcounter += 1
        return itertools.chain(gen, gen2)

    def _format_body(self) -> Iterable[ExcelCell]:
        if isinstance(self.df.index, MultiIndex):
            return self._format_hierarchical_rows()
        else:
            return self._format_regular_rows()

    def _format_regular_rows(self) -> Iterable[ExcelCell]:
        if self._has_aliases or self.header:
            self.rowcounter += 1
        if self.index:
            if self.index_label and isinstance(self.index_label, (list, tuple, np.ndarray, Index)):
                index_label = self.index_label[0]
            elif self.index_label and isinstance(self.index_label, str):
                index_label = self.index_label
            else:
                index_label = self.df.index.names[0]
            if isinstance(self.columns, MultiIndex):
                self.rowcounter += 1
            if index_label and self.header is not False:
                yield ExcelCell(self.rowcounter - 1, 0, index_label, self.header_style)
            index_values = self.df.index
            if isinstance(self.df.index, PeriodIndex):
                index_values = self.df.index.to_timestamp()
            for idx, idxval in enumerate(index_values):
                yield CssExcelCell(row=self.rowcounter + idx, col=0, val=idxval, style=self.header_style, css_styles=getattr(self.styler, 'ctx_index', None), css_row=idx, css_col=0, css_converter=self.style_converter)
            coloffset = 1
        else:
            coloffset = 0
        yield from self._generate_body(coloffset)

    def _format_hierarchical_rows(self) -> Iterable[ExcelCell]:
        if self._has_aliases or self.header:
            self.rowcounter += 1
        gcolidx = 0
        if self.index:
            index_labels = self.df.index.names
            if self.index_label and isinstance(self.index_label, (list, tuple, np.ndarray, Index)):
                index_labels = self.index_label
            if isinstance(self.columns, MultiIndex) and self.merge_cells:
                self.rowcounter += 1
            if com.any_not_none(*index_labels) and self.header is not False:
                for cidx, name in enumerate(index_labels):
                    yield ExcelCell(self.rowcounter - 1, cidx, name, self.header_style)
            if self.merge_cells:
                level_strs = self.df.index._format_multi(sparsify=True, include_names=False)
                level_lengths = get_level_lengths(level_strs)
                for spans, levels, level_codes in zip(level_lengths, self.df.index.levels, self.df.index.codes):
                    values = levels.take(level_codes, allow_fill=levels._can_hold_na, fill_value=levels._na_value)
                    for i, span_val in spans.items():
                        mergestart, mergeend = (None, None)
                        if span_val > 1:
                            mergestart = self.rowcounter + i + span_val - 1
                            mergeend = gcolidx
                        yield CssExcelCell(row=self.rowcounter + i, col=gcolidx, val=values[i], style=self.header_style, css_styles=getattr(self.styler, 'ctx_index', None), css_row=i, css_col=gcolidx, css_converter=self.style_converter, mergestart=mergestart, mergeend=mergeend)
                    gcolidx += 1
            else:
                for indexcolvals in zip(*self.df.index):
                    for idx, indexcolval in enumerate(indexcolvals):
                        yield CssExcelCell(row=self.rowcounter + idx, col=gcolidx, val=indexcolval, style=self.header_style, css_styles=getattr(self.styler, 'ctx_index', None), css_row=idx, css_col=gcolidx, css_converter=self.style_converter)
                    gcolidx += 1
        yield from self._generate_body(gcolidx)

    @property
    def _has_aliases(self) -> bool:
        """Whether the aliases for column names are present."""
        return is_list_like(self.header)

    def _generate_body(self, coloffset: int) -> Iterable[ExcelCell]:
        for colidx in range(len(self.columns)):
            series = self.df.iloc[:, colidx]
            for i, val in enumerate(series):
                yield CssExcelCell(row=self.rowcounter + i, col=colidx + coloffset, val=val, style=None, css_styles=getattr(self.styler, 'ctx', None), css_row=i, css_col=colidx, css_converter=self.style_converter)

    def get_formatted_cells(self) -> Iterable[ExcelCell]:
        for cell in itertools.chain(self._format_header(), self._format_body()):
            cell.val = self._format_value(cell.val)
            yield cell

    @doc(storage_options=_shared_docs['storage_options'])
    def write(self, writer: FilePath | WriteExcelBuffer | ExcelWriter, sheet_name: str='Sheet1', startrow: int=0, startcol: int=0, freeze_panes: tuple[int, int] | None=None, engine: str | None=None, storage_options: StorageOptions | None=None, engine_kwargs: dict | None=None) -> None:
        """
        writer : path-like, file-like, or ExcelWriter object
            File path or existing ExcelWriter
        sheet_name : str, default 'Sheet1'
            Name of sheet which will contain DataFrame
        startrow :
            upper left cell row to dump data frame
        startcol :
            upper left cell column to dump data frame
        freeze_panes : tuple of integer (length 2), default None
            Specifies the one-based bottommost row and rightmost column that
            is to be frozen
        engine : string, default None
            write engine to use if writer is a path - you can also set this
            via the options ``io.excel.xlsx.writer``,
            or ``io.excel.xlsm.writer``.

        {storage_options}

        engine_kwargs: dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        from pandas.io.excel import ExcelWriter
        num_rows, num_cols = self.df.shape
        if num_rows > self.max_rows or num_cols > self.max_cols:
            raise ValueError(f'This sheet is too large! Your sheet size is: {num_rows}, {num_cols} Max sheet size is: {self.max_rows}, {self.max_cols}')
        if engine_kwargs is None:
            engine_kwargs = {}
        formatted_cells = self.get_formatted_cells()
        if isinstance(writer, ExcelWriter):
            need_save = False
        else:
            writer = ExcelWriter(writer, engine=engine, storage_options=storage_options, engine_kwargs=engine_kwargs)
            need_save = True
        try:
            writer._write_cells(formatted_cells, sheet_name, startrow=startrow, startcol=startcol, freeze_panes=freeze_panes)
        finally:
            if need_save:
                writer.close()
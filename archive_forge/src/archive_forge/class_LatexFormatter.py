from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
class LatexFormatter:
    """
    Used to render a DataFrame to a LaTeX tabular/longtable environment output.

    Parameters
    ----------
    formatter : `DataFrameFormatter`
    longtable : bool, default False
        Use longtable environment.
    column_format : str, default None
        The columns format as specified in `LaTeX table format
        <https://en.wikibooks.org/wiki/LaTeX/Tables>`__ e.g 'rcl' for 3 columns
    multicolumn : bool, default False
        Use \\multicolumn to enhance MultiIndex columns.
    multicolumn_format : str, default 'l'
        The alignment for multicolumns, similar to `column_format`
    multirow : bool, default False
        Use \\multirow to enhance MultiIndex rows.
    caption : str or tuple, optional
        Tuple (full_caption, short_caption),
        which results in \\caption[short_caption]{full_caption};
        if a single string is passed, no short caption will be set.
    label : str, optional
        The LaTeX label to be placed inside ``\\label{}`` in the output.
    position : str, optional
        The LaTeX positional argument for tables, to be placed after
        ``\\begin{}`` in the output.

    See Also
    --------
    HTMLFormatter
    """

    def __init__(self, formatter: DataFrameFormatter, longtable: bool=False, column_format: str | None=None, multicolumn: bool=False, multicolumn_format: str | None=None, multirow: bool=False, caption: str | tuple[str, str] | None=None, label: str | None=None, position: str | None=None) -> None:
        self.fmt = formatter
        self.frame = self.fmt.frame
        self.longtable = longtable
        self.column_format = column_format
        self.multicolumn = multicolumn
        self.multicolumn_format = multicolumn_format
        self.multirow = multirow
        self.caption, self.short_caption = _split_into_full_short_caption(caption)
        self.label = label
        self.position = position

    def to_string(self) -> str:
        """
        Render a DataFrame to a LaTeX tabular, longtable, or table/tabular
        environment output.
        """
        return self.builder.get_result()

    @property
    def builder(self) -> TableBuilderAbstract:
        """Concrete table builder.

        Returns
        -------
        TableBuilder
        """
        builder = self._select_builder()
        return builder(formatter=self.fmt, column_format=self.column_format, multicolumn=self.multicolumn, multicolumn_format=self.multicolumn_format, multirow=self.multirow, caption=self.caption, short_caption=self.short_caption, label=self.label, position=self.position)

    def _select_builder(self) -> type[TableBuilderAbstract]:
        """Select proper table builder."""
        if self.longtable:
            return LongTableBuilder
        if any([self.caption, self.label, self.position]):
            return RegularTableBuilder
        return TabularBuilder

    @property
    def column_format(self) -> str | None:
        """Column format."""
        return self._column_format

    @column_format.setter
    def column_format(self, input_column_format: str | None) -> None:
        """Setter for column format."""
        if input_column_format is None:
            self._column_format = self._get_index_format() + self._get_column_format_based_on_dtypes()
        elif not isinstance(input_column_format, str):
            raise ValueError(f'column_format must be str or unicode, not {type(input_column_format)}')
        else:
            self._column_format = input_column_format

    def _get_column_format_based_on_dtypes(self) -> str:
        """Get column format based on data type.

        Right alignment for numbers and left - for strings.
        """

        def get_col_type(dtype) -> str:
            if issubclass(dtype.type, np.number):
                return 'r'
            return 'l'
        dtypes = self.frame.dtypes._values
        return ''.join(map(get_col_type, dtypes))

    def _get_index_format(self) -> str:
        """Get index column format."""
        return 'l' * self.frame.index.nlevels if self.fmt.index else ''
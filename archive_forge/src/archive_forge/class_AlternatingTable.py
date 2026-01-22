import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
class AlternatingTable(BorderedTable):
    """
    Implementation of BorderedTable which uses background colors to distinguish between rows instead of row border
    lines. This class can be used to create the whole table at once or one row at a time.

    To nest an AlternatingTable within another AlternatingTable, set style_data_text to False on the Column
    which contains the nested table. That will prevent the current row's background color from affecting the colors
    of the nested table.
    """

    def __init__(self, cols: Sequence[Column], *, tab_width: int=4, column_borders: bool=True, padding: int=1, border_fg: Optional[ansi.FgColor]=None, border_bg: Optional[ansi.BgColor]=None, header_bg: Optional[ansi.BgColor]=None, odd_bg: Optional[ansi.BgColor]=None, even_bg: Optional[ansi.BgColor]=ansi.Bg.DARK_GRAY) -> None:
        """
        AlternatingTable initializer

        Note: Specify background colors using subclasses of BgColor (e.g. Bg, EightBitBg, RgbBg)

        :param cols: column definitions for this table
        :param tab_width: all tabs will be replaced with this many spaces. If a row's fill_char is a tab,
                          then it will be converted to one space.
        :param column_borders: if True, borders between columns will be included. This gives the table a grid-like
                               appearance. Turning off column borders results in a unified appearance between
                               a row's cells. (Defaults to True)
        :param padding: number of spaces between text and left/right borders of cell
        :param border_fg: optional foreground color for borders (defaults to None)
        :param border_bg: optional background color for borders (defaults to None)
        :param header_bg: optional background color for header cells (defaults to None)
        :param odd_bg: optional background color for odd numbered data rows (defaults to None)
        :param even_bg: optional background color for even numbered data rows (defaults to StdBg.DARK_GRAY)
        :raises: ValueError if tab_width is less than 1
        :raises: ValueError if padding is less than 0
        """
        super().__init__(cols, tab_width=tab_width, column_borders=column_borders, padding=padding, border_fg=border_fg, border_bg=border_bg, header_bg=header_bg)
        self.row_num = 1
        self.odd_bg = odd_bg
        self.even_bg = even_bg

    def apply_data_bg(self, value: Any) -> str:
        """
        Apply background color to data text based on what row is being generated and whether a color has been defined
        :param value: object whose text is to be colored
        :return: formatted data string
        """
        if self.row_num % 2 == 0 and self.even_bg is not None:
            return ansi.style(value, bg=self.even_bg)
        elif self.row_num % 2 != 0 and self.odd_bg is not None:
            return ansi.style(value, bg=self.odd_bg)
        else:
            return str(value)

    def generate_data_row(self, row_data: Sequence[Any]) -> str:
        """
        Generate a data row

        :param row_data: data with an entry for each column in the row
        :return: data row string
        """
        row = super().generate_data_row(row_data)
        self.row_num += 1
        return row

    def generate_table(self, table_data: Sequence[Sequence[Any]], *, include_header: bool=True) -> str:
        """
        Generate a table from a data set

        :param table_data: Data with an entry for each data row of the table. Each entry should have data for
                           each column in the row.
        :param include_header: If True, then a header will be included at top of table. (Defaults to True)
        """
        table_buf = io.StringIO()
        if include_header:
            header = self.generate_header()
            table_buf.write(header)
        else:
            top_border = self.generate_table_top_border()
            table_buf.write(top_border)
        table_buf.write('\n')
        for row_data in table_data:
            row = self.generate_data_row(row_data)
            table_buf.write(row)
            table_buf.write('\n')
        table_buf.write(self.generate_table_bottom_border())
        return table_buf.getvalue()
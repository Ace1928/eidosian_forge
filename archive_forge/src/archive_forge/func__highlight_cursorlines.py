from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from .controls import UIControl, TokenListControl, UIContent
from .dimension import LayoutDimension, sum_layout_dimensions, max_layout_dimensions
from .margins import Margin
from .screen import Point, WritePosition, _CHAR_CACHE
from .utils import token_list_to_text, explode_tokens
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import to_cli_filter, ViInsertMode, EmacsInsertMode
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.reactive import Integer
from prompt_toolkit.token import Token
from prompt_toolkit.utils import take_using_weights, get_cwidth
def _highlight_cursorlines(self, cli, new_screen, cpos, x, y, width, height):
    """
        Highlight cursor row/column.
        """
    cursor_line_token = (':',) + self.cursorline_token
    cursor_column_token = (':',) + self.cursorcolumn_token
    data_buffer = new_screen.data_buffer
    if self.cursorline(cli):
        row = data_buffer[cpos.y]
        for x in range(x, x + width):
            original_char = row[x]
            row[x] = _CHAR_CACHE[original_char.char, original_char.token + cursor_line_token]
    if self.cursorcolumn(cli):
        for y2 in range(y, y + height):
            row = data_buffer[y2]
            original_char = row[cpos.x]
            row[cpos.x] = _CHAR_CACHE[original_char.char, original_char.token + cursor_column_token]
    for cc in self.get_colorcolumns(cli):
        assert isinstance(cc, ColorColumn)
        color_column_token = (':',) + cc.token
        column = cc.position
        for y2 in range(y, y + height):
            row = data_buffer[y2]
            original_char = row[column]
            row[column] = _CHAR_CACHE[original_char.char, original_char.token + color_column_token]
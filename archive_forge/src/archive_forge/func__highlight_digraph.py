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
def _highlight_digraph(self, cli, new_screen):
    """
        When we are in Vi digraph mode, put a question mark underneath the
        cursor.
        """
    digraph_char = self._get_digraph_char(cli)
    if digraph_char:
        cpos = new_screen.cursor_position
        new_screen.data_buffer[cpos.y][cpos.x] = _CHAR_CACHE[digraph_char, Token.Digraph]
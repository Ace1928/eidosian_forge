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
def _get_margin_width(self, cli, margin):
    """
        Return the width for this margin.
        (Calculate only once per render time.)
        """

    def get_ui_content():
        return self._get_ui_content(cli, width=0, height=0)

    def get_width():
        return margin.get_width(cli, get_ui_content)
    key = (margin, cli.render_counter)
    return self._margin_width_cache.get(key, get_width)
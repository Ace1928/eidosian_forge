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
@property
def applied_scroll_offsets(self):
    """
        Return a :class:`.ScrollOffsets` instance that indicates the actual
        offset. This can be less than or equal to what's configured. E.g, when
        the cursor is completely at the top, the top offset will be zero rather
        than what's configured.
        """
    if self.displayed_lines[0] == 0:
        top = 0
    else:
        y = self.input_line_to_visible_line[self.ui_content.cursor_position.y]
        top = min(y, self.configured_scroll_offsets.top)
    return ScrollOffsets(top=top, bottom=min(self.ui_content.line_count - self.displayed_lines[-1] - 1, self.configured_scroll_offsets.bottom), left=0, right=0)
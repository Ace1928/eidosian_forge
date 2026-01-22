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
class ScrollOffsets(object):
    """
    Scroll offsets for the :class:`.Window` class.

    Note that left/right offsets only make sense if line wrapping is disabled.
    """

    def __init__(self, top=0, bottom=0, left=0, right=0):
        assert isinstance(top, Integer)
        assert isinstance(bottom, Integer)
        assert isinstance(left, Integer)
        assert isinstance(right, Integer)
        self._top = top
        self._bottom = bottom
        self._left = left
        self._right = right

    @property
    def top(self):
        return int(self._top)

    @property
    def bottom(self):
        return int(self._bottom)

    @property
    def left(self):
        return int(self._left)

    @property
    def right(self):
        return int(self._right)

    def __repr__(self):
        return 'ScrollOffsets(top=%r, bottom=%r, left=%r, right=%r)' % (self.top, self.bottom, self.left, self.right)
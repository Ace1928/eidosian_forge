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
@staticmethod
def _merge_dimensions(dimension, preferred=None, dont_extend=False):
    """
        Take the LayoutDimension from this `Window` class and the received
        preferred size from the `UIControl` and return a `LayoutDimension` to
        report to the parent container.
        """
    dimension = dimension or LayoutDimension()
    if dimension.preferred_specified:
        preferred = dimension.preferred
    if preferred is not None:
        if dimension.max:
            preferred = min(preferred, dimension.max)
        if dimension.min:
            preferred = max(preferred, dimension.min)
    if dont_extend and preferred is not None:
        max_ = min(dimension.max, preferred)
    else:
        max_ = dimension.max
    return LayoutDimension(min=dimension.min, max=max_, preferred=preferred, weight=dimension.weight)
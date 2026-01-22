from __future__ import annotations
import math
from itertools import zip_longest
from typing import TYPE_CHECKING, Callable, Iterable, Sequence, TypeVar, cast
from weakref import WeakKeyDictionary
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import CompletionState
from prompt_toolkit.completion import Completion
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth
from .containers import ConditionalContainer, HSplit, ScrollOffsets, Window
from .controls import GetLinePrefixCallable, UIContent, UIControl
from .dimension import Dimension
from .margins import ScrollbarMargin
def _get_menu_item_fragments(completion: Completion, is_current_completion: bool, width: int, space_after: bool=False) -> StyleAndTextTuples:
    """
    Get the style/text tuples for a menu item, styled and trimmed to the given
    width.
    """
    if is_current_completion:
        style_str = 'class:completion-menu.completion.current {} {}'.format(completion.style, completion.selected_style)
    else:
        style_str = 'class:completion-menu.completion ' + completion.style
    text, tw = _trim_formatted_text(completion.display, width - 2 if space_after else width - 1)
    padding = ' ' * (width - 1 - tw)
    return to_formatted_text(cast(StyleAndTextTuples, []) + [('', ' ')] + text + [('', padding)], style=style_str)
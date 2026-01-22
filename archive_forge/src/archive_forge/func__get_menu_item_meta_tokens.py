from __future__ import unicode_literals
from six.moves import zip_longest, range
from prompt_toolkit.filters import HasCompletions, IsDone, Condition, to_cli_filter
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .containers import Window, HSplit, ConditionalContainer, ScrollOffsets
from .controls import UIControl, UIContent
from .dimension import LayoutDimension
from .margins import ScrollbarMargin
from .screen import Point, Char
import math
def _get_menu_item_meta_tokens(self, completion, is_current_completion, width):
    if is_current_completion:
        token = self.token.Meta.Current
    else:
        token = self.token.Meta
    text, tw = _trim_text(completion.display_meta, width - 2)
    padding = ' ' * (width - 2 - tw)
    return [(token, ' %s%s ' % (text, padding))]
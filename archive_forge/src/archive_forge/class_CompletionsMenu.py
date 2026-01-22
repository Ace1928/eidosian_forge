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
class CompletionsMenu(ConditionalContainer):

    def __init__(self, max_height=None, scroll_offset=0, extra_filter=True, display_arrows=False):
        extra_filter = to_cli_filter(extra_filter)
        display_arrows = to_cli_filter(display_arrows)
        super(CompletionsMenu, self).__init__(content=Window(content=CompletionsMenuControl(), width=LayoutDimension(min=8), height=LayoutDimension(min=1, max=max_height), scroll_offsets=ScrollOffsets(top=scroll_offset, bottom=scroll_offset), right_margins=[ScrollbarMargin(display_arrows=display_arrows)], dont_extend_width=True), filter=HasCompletions() & ~IsDone() & extra_filter)
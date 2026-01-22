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
class CompletionsMenuControl(UIControl):
    """
    Helper for drawing the complete menu to the screen.

    :param scroll_offset: Number (integer) representing the preferred amount of
        completions to be displayed before and after the current one. When this
        is a very high number, the current completion will be shown in the
        middle most of the time.
    """
    MIN_WIDTH = 7

    def __init__(self):
        self.token = Token.Menu.Completions

    def has_focus(self, cli):
        return False

    def preferred_width(self, cli, max_available_width):
        complete_state = cli.current_buffer.complete_state
        if complete_state:
            menu_width = self._get_menu_width(500, complete_state)
            menu_meta_width = self._get_menu_meta_width(500, complete_state)
            return menu_width + menu_meta_width
        else:
            return 0

    def preferred_height(self, cli, width, max_available_height, wrap_lines):
        complete_state = cli.current_buffer.complete_state
        if complete_state:
            return len(complete_state.current_completions)
        else:
            return 0

    def create_content(self, cli, width, height):
        """
        Create a UIContent object for this control.
        """
        complete_state = cli.current_buffer.complete_state
        if complete_state:
            completions = complete_state.current_completions
            index = complete_state.complete_index
            menu_width = self._get_menu_width(width, complete_state)
            menu_meta_width = self._get_menu_meta_width(width - menu_width, complete_state)
            show_meta = self._show_meta(complete_state)

            def get_line(i):
                c = completions[i]
                is_current_completion = i == index
                result = self._get_menu_item_tokens(c, is_current_completion, menu_width)
                if show_meta:
                    result += self._get_menu_item_meta_tokens(c, is_current_completion, menu_meta_width)
                return result
            return UIContent(get_line=get_line, cursor_position=Point(x=0, y=index or 0), line_count=len(completions), default_char=Char(' ', self.token))
        return UIContent()

    def _show_meta(self, complete_state):
        """
        Return ``True`` if we need to show a column with meta information.
        """
        return any((c.display_meta for c in complete_state.current_completions))

    def _get_menu_width(self, max_width, complete_state):
        """
        Return the width of the main column.
        """
        return min(max_width, max(self.MIN_WIDTH, max((get_cwidth(c.display) for c in complete_state.current_completions)) + 2))

    def _get_menu_meta_width(self, max_width, complete_state):
        """
        Return the width of the meta column.
        """
        if self._show_meta(complete_state):
            return min(max_width, max((get_cwidth(c.display_meta) for c in complete_state.current_completions)) + 2)
        else:
            return 0

    def _get_menu_item_tokens(self, completion, is_current_completion, width):
        if is_current_completion:
            token = self.token.Completion.Current
        else:
            token = self.token.Completion
        text, tw = _trim_text(completion.display, width - 2)
        padding = ' ' * (width - 2 - tw)
        return [(token, ' %s%s ' % (text, padding))]

    def _get_menu_item_meta_tokens(self, completion, is_current_completion, width):
        if is_current_completion:
            token = self.token.Meta.Current
        else:
            token = self.token.Meta
        text, tw = _trim_text(completion.display_meta, width - 2)
        padding = ' ' * (width - 2 - tw)
        return [(token, ' %s%s ' % (text, padding))]

    def mouse_handler(self, cli, mouse_event):
        """
        Handle mouse events: clicking and scrolling.
        """
        b = cli.current_buffer
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            b.go_to_completion(mouse_event.position.y)
            b.complete_state = None
        elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            b.complete_next(count=3, disable_wrap_around=True)
        elif mouse_event.event_type == MouseEventType.SCROLL_UP:
            b.complete_previous(count=3, disable_wrap_around=True)
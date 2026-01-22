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
class MultiColumnCompletionMenuControl(UIControl):
    """
    Completion menu that displays all the completions in several columns.
    When there are more completions than space for them to be displayed, an
    arrow is shown on the left or right side.

    `min_rows` indicates how many rows will be available in any possible case.
    When this is langer than one, in will try to use less columns and more
    rows until this value is reached.
    Be careful passing in a too big value, if less than the given amount of
    rows are available, more columns would have been required, but
    `preferred_width` doesn't know about that and reports a too small value.
    This results in less completions displayed and additional scrolling.
    (It's a limitation of how the layout engine currently works: first the
    widths are calculated, then the heights.)

    :param suggested_max_column_width: The suggested max width of a column.
        The column can still be bigger than this, but if there is place for two
        columns of this width, we will display two columns. This to avoid that
        if there is one very wide completion, that it doesn't significantly
        reduce the amount of columns.
    """
    _required_margin = 3

    def __init__(self, min_rows=3, suggested_max_column_width=30):
        assert isinstance(min_rows, int) and min_rows >= 1
        self.min_rows = min_rows
        self.suggested_max_column_width = suggested_max_column_width
        self.token = Token.Menu.Completions
        self.scroll = 0
        self._rendered_rows = 0
        self._rendered_columns = 0
        self._total_columns = 0
        self._render_pos_to_completion = {}
        self._render_left_arrow = False
        self._render_right_arrow = False
        self._render_width = 0

    def reset(self):
        self.scroll = 0

    def has_focus(self, cli):
        return False

    def preferred_width(self, cli, max_available_width):
        """
        Preferred width: prefer to use at least min_rows, but otherwise as much
        as possible horizontally.
        """
        complete_state = cli.current_buffer.complete_state
        column_width = self._get_column_width(complete_state)
        result = int(column_width * math.ceil(len(complete_state.current_completions) / float(self.min_rows)))
        while result > column_width and result > max_available_width - self._required_margin:
            result -= column_width
        return result + self._required_margin

    def preferred_height(self, cli, width, max_available_height, wrap_lines):
        """
        Preferred height: as much as needed in order to display all the completions.
        """
        complete_state = cli.current_buffer.complete_state
        column_width = self._get_column_width(complete_state)
        column_count = max(1, (width - self._required_margin) // column_width)
        return int(math.ceil(len(complete_state.current_completions) / float(column_count)))

    def create_content(self, cli, width, height):
        """
        Create a UIContent object for this menu.
        """
        complete_state = cli.current_buffer.complete_state
        column_width = self._get_column_width(complete_state)
        self._render_pos_to_completion = {}

        def grouper(n, iterable, fillvalue=None):
            """ grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx """
            args = [iter(iterable)] * n
            return zip_longest(*args, fillvalue=fillvalue)

        def is_current_completion(completion):
            """ Returns True when this completion is the currently selected one. """
            return complete_state.complete_index is not None and c == complete_state.current_completion
        HORIZONTAL_MARGIN_REQUIRED = 3
        if complete_state:
            column_width = min(width - HORIZONTAL_MARGIN_REQUIRED, column_width)
            if column_width > self.suggested_max_column_width:
                column_width //= column_width // self.suggested_max_column_width
            visible_columns = max(1, (width - self._required_margin) // column_width)
            columns_ = list(grouper(height, complete_state.current_completions))
            rows_ = list(zip(*columns_))
            selected_column = (complete_state.complete_index or 0) // height
            self.scroll = min(selected_column, max(self.scroll, selected_column - visible_columns + 1))
            render_left_arrow = self.scroll > 0
            render_right_arrow = self.scroll < len(rows_[0]) - visible_columns
            tokens_for_line = []
            for row_index, row in enumerate(rows_):
                tokens = []
                middle_row = row_index == len(rows_) // 2
                if render_left_arrow:
                    tokens += [(Token.Scrollbar, '<' if middle_row else ' ')]
                for column_index, c in enumerate(row[self.scroll:][:visible_columns]):
                    if c is not None:
                        tokens += self._get_menu_item_tokens(c, is_current_completion(c), column_width)
                        for x in range(column_width):
                            self._render_pos_to_completion[column_index * column_width + x, row_index] = c
                    else:
                        tokens += [(self.token.Completion, ' ' * column_width)]
                tokens += [(self.token.Completion, ' ')]
                if render_right_arrow:
                    tokens += [(Token.Scrollbar, '>' if middle_row else ' ')]
                tokens_for_line.append(tokens)
        else:
            tokens = []
        self._rendered_rows = height
        self._rendered_columns = visible_columns
        self._total_columns = len(columns_)
        self._render_left_arrow = render_left_arrow
        self._render_right_arrow = render_right_arrow
        self._render_width = column_width * visible_columns + render_left_arrow + render_right_arrow + 1

        def get_line(i):
            return tokens_for_line[i]
        return UIContent(get_line=get_line, line_count=len(rows_))

    def _get_column_width(self, complete_state):
        """
        Return the width of each column.
        """
        return max((get_cwidth(c.display) for c in complete_state.current_completions)) + 1

    def _get_menu_item_tokens(self, completion, is_current_completion, width):
        if is_current_completion:
            token = self.token.Completion.Current
        else:
            token = self.token.Completion
        text, tw = _trim_text(completion.display, width)
        padding = ' ' * (width - tw - 1)
        return [(token, ' %s%s' % (text, padding))]

    def mouse_handler(self, cli, mouse_event):
        """
        Handle scoll and click events.
        """
        b = cli.current_buffer

        def scroll_left():
            b.complete_previous(count=self._rendered_rows, disable_wrap_around=True)
            self.scroll = max(0, self.scroll - 1)

        def scroll_right():
            b.complete_next(count=self._rendered_rows, disable_wrap_around=True)
            self.scroll = min(self._total_columns - self._rendered_columns, self.scroll + 1)
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            scroll_right()
        elif mouse_event.event_type == MouseEventType.SCROLL_UP:
            scroll_left()
        elif mouse_event.event_type == MouseEventType.MOUSE_UP:
            x = mouse_event.position.x
            y = mouse_event.position.y
            if x == 0:
                if self._render_left_arrow:
                    scroll_left()
            elif x == self._render_width - 1:
                if self._render_right_arrow:
                    scroll_right()
            else:
                completion = self._render_pos_to_completion.get((x, y))
                if completion:
                    b.apply_completion(completion)
from __future__ import annotations
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.key_binding import KeyBindingsBase
from prompt_toolkit.mouse_events import MouseEvent
from .containers import Container, ScrollOffsets
from .dimension import AnyDimension, Dimension, sum_layout_dimensions, to_dimension
from .mouse_handlers import MouseHandler, MouseHandlers
from .screen import Char, Screen, WritePosition
def _draw_scrollbar(self, write_position: WritePosition, content_height: int, screen: Screen) -> None:
    """
        Draw the scrollbar on the screen.

        Note: There is some code duplication with the `ScrollbarMargin`
              implementation.
        """
    window_height = write_position.height
    display_arrows = self.display_arrows()
    if display_arrows:
        window_height -= 2
    try:
        fraction_visible = write_position.height / float(content_height)
        fraction_above = self.vertical_scroll / float(content_height)
        scrollbar_height = int(min(window_height, max(1, window_height * fraction_visible)))
        scrollbar_top = int(window_height * fraction_above)
    except ZeroDivisionError:
        return
    else:

        def is_scroll_button(row: int) -> bool:
            """True if we should display a button on this row."""
            return scrollbar_top <= row <= scrollbar_top + scrollbar_height
        xpos = write_position.xpos + write_position.width - 1
        ypos = write_position.ypos
        data_buffer = screen.data_buffer
        if display_arrows:
            data_buffer[ypos][xpos] = Char(self.up_arrow_symbol, 'class:scrollbar.arrow')
            ypos += 1
        scrollbar_background = 'class:scrollbar.background'
        scrollbar_background_start = 'class:scrollbar.background,scrollbar.start'
        scrollbar_button = 'class:scrollbar.button'
        scrollbar_button_end = 'class:scrollbar.button,scrollbar.end'
        for i in range(window_height):
            style = ''
            if is_scroll_button(i):
                if not is_scroll_button(i + 1):
                    style = scrollbar_button_end
                else:
                    style = scrollbar_button
            elif is_scroll_button(i + 1):
                style = scrollbar_background_start
            else:
                style = scrollbar_background
            data_buffer[ypos][xpos] = Char(' ', style)
            ypos += 1
        if display_arrows:
            data_buffer[ypos][xpos] = Char(self.down_arrow_symbol, 'class:scrollbar.arrow')
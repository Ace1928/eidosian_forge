from __future__ import annotations
from abc import ABCMeta, abstractmethod
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Callable, Sequence, Union, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import (
from prompt_toolkit.key_binding import KeyBindingsBase
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth, take_using_weights, to_int, to_str
from .controls import (
from .dimension import (
from .margins import Margin
from .mouse_handlers import MouseHandlers
from .screen import _CHAR_CACHE, Screen, WritePosition
from .utils import explode_text_fragments
def _write_to_screen_at_index(self, screen: Screen, mouse_handlers: MouseHandlers, write_position: WritePosition, parent_style: str, erase_bg: bool) -> None:
    if write_position.height <= 0 or write_position.width <= 0:
        return
    left_margin_widths = [self._get_margin_width(m) for m in self.left_margins]
    right_margin_widths = [self._get_margin_width(m) for m in self.right_margins]
    total_margin_width = sum(left_margin_widths + right_margin_widths)
    ui_content = self.content.create_content(write_position.width - total_margin_width, write_position.height)
    assert isinstance(ui_content, UIContent)
    wrap_lines = self.wrap_lines()
    self._scroll(ui_content, write_position.width - total_margin_width, write_position.height)
    self._fill_bg(screen, write_position, erase_bg)
    align = self.align() if callable(self.align) else self.align
    visible_line_to_row_col, rowcol_to_yx = self._copy_body(ui_content, screen, write_position, sum(left_margin_widths), write_position.width - total_margin_width, self.vertical_scroll, self.horizontal_scroll, wrap_lines=wrap_lines, highlight_lines=True, vertical_scroll_2=self.vertical_scroll_2, always_hide_cursor=self.always_hide_cursor(), has_focus=get_app().layout.current_control == self.content, align=align, get_line_prefix=self.get_line_prefix)
    x_offset = write_position.xpos + sum(left_margin_widths)
    y_offset = write_position.ypos
    render_info = WindowRenderInfo(window=self, ui_content=ui_content, horizontal_scroll=self.horizontal_scroll, vertical_scroll=self.vertical_scroll, window_width=write_position.width - total_margin_width, window_height=write_position.height, configured_scroll_offsets=self.scroll_offsets, visible_line_to_row_col=visible_line_to_row_col, rowcol_to_yx=rowcol_to_yx, x_offset=x_offset, y_offset=y_offset, wrap_lines=wrap_lines)
    self.render_info = render_info

    def mouse_handler(mouse_event: MouseEvent) -> NotImplementedOrNone:
        """
            Wrapper around the mouse_handler of the `UIControl` that turns
            screen coordinates into line coordinates.
            Returns `NotImplemented` if no UI invalidation should be done.
            """
        if self not in get_app().layout.walk_through_modal_area():
            return NotImplemented
        yx_to_rowcol = {v: k for k, v in rowcol_to_yx.items()}
        y = mouse_event.position.y
        x = mouse_event.position.x
        max_y = write_position.ypos + len(visible_line_to_row_col) - 1
        y = min(max_y, y)
        result: NotImplementedOrNone
        while x >= 0:
            try:
                row, col = yx_to_rowcol[y, x]
            except KeyError:
                x -= 1
            else:
                result = self.content.mouse_handler(MouseEvent(position=Point(x=col, y=row), event_type=mouse_event.event_type, button=mouse_event.button, modifiers=mouse_event.modifiers))
                break
        else:
            result = self.content.mouse_handler(MouseEvent(position=Point(x=0, y=0), event_type=mouse_event.event_type, button=mouse_event.button, modifiers=mouse_event.modifiers))
        if result == NotImplemented:
            result = self._mouse_handler(mouse_event)
        return result
    mouse_handlers.set_mouse_handler_for_range(x_min=write_position.xpos + sum(left_margin_widths), x_max=write_position.xpos + write_position.width - total_margin_width, y_min=write_position.ypos, y_max=write_position.ypos + write_position.height, handler=mouse_handler)
    move_x = 0

    def render_margin(m: Margin, width: int) -> UIContent:
        """Render margin. Return `Screen`."""
        fragments = m.create_margin(render_info, width, write_position.height)
        return FormattedTextControl(fragments).create_content(width + 1, write_position.height)
    for m, width in zip(self.left_margins, left_margin_widths):
        if width > 0:
            margin_content = render_margin(m, width)
            self._copy_margin(margin_content, screen, write_position, move_x, width)
            move_x += width
    move_x = write_position.width - sum(right_margin_widths)
    for m, width in zip(self.right_margins, right_margin_widths):
        margin_content = render_margin(m, width)
        self._copy_margin(margin_content, screen, write_position, move_x, width)
        move_x += width
    self._apply_style(screen, write_position, parent_style)
    screen.visible_windows_to_write_positions[self] = write_position
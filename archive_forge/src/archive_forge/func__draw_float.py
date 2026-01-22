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
def _draw_float(self, fl: Float, screen: Screen, mouse_handlers: MouseHandlers, write_position: WritePosition, style: str, erase_bg: bool, z_index: int | None) -> None:
    """Draw a single Float."""
    cpos = screen.get_menu_position(fl.attach_to_window or get_app().layout.current_window)
    cursor_position = Point(x=cpos.x - write_position.xpos, y=cpos.y - write_position.ypos)
    fl_width = fl.get_width()
    fl_height = fl.get_height()
    width: int
    height: int
    xpos: int
    ypos: int
    if fl.left is not None and fl_width is not None:
        xpos = fl.left
        width = fl_width
    elif fl.left is not None and fl.right is not None:
        xpos = fl.left
        width = write_position.width - fl.left - fl.right
    elif fl_width is not None and fl.right is not None:
        xpos = write_position.width - fl.right - fl_width
        width = fl_width
    elif fl.xcursor:
        if fl_width is None:
            width = fl.content.preferred_width(write_position.width).preferred
            width = min(write_position.width, width)
        else:
            width = fl_width
        xpos = cursor_position.x
        if xpos + width > write_position.width:
            xpos = max(0, write_position.width - width)
    elif fl_width:
        xpos = int((write_position.width - fl_width) / 2)
        width = fl_width
    else:
        width = fl.content.preferred_width(write_position.width).preferred
        if fl.left is not None:
            xpos = fl.left
        elif fl.right is not None:
            xpos = max(0, write_position.width - width - fl.right)
        else:
            xpos = max(0, int((write_position.width - width) / 2))
        width = min(width, write_position.width - xpos)
    if fl.top is not None and fl_height is not None:
        ypos = fl.top
        height = fl_height
    elif fl.top is not None and fl.bottom is not None:
        ypos = fl.top
        height = write_position.height - fl.top - fl.bottom
    elif fl_height is not None and fl.bottom is not None:
        ypos = write_position.height - fl_height - fl.bottom
        height = fl_height
    elif fl.ycursor:
        ypos = cursor_position.y + (0 if fl.allow_cover_cursor else 1)
        if fl_height is None:
            height = fl.content.preferred_height(width, write_position.height).preferred
        else:
            height = fl_height
        if height > write_position.height - ypos:
            if write_position.height - ypos + 1 >= ypos:
                height = write_position.height - ypos
            else:
                height = min(height, cursor_position.y)
                ypos = cursor_position.y - height
    elif fl_height:
        ypos = int((write_position.height - fl_height) / 2)
        height = fl_height
    else:
        height = fl.content.preferred_height(width, write_position.height).preferred
        if fl.top is not None:
            ypos = fl.top
        elif fl.bottom is not None:
            ypos = max(0, write_position.height - height - fl.bottom)
        else:
            ypos = max(0, int((write_position.height - height) / 2))
        height = min(height, write_position.height - ypos)
    if height > 0 and width > 0:
        wp = WritePosition(xpos=xpos + write_position.xpos, ypos=ypos + write_position.ypos, width=width, height=height)
        if not fl.hide_when_covering_content or self._area_is_empty(screen, wp):
            fl.content.write_to_screen(screen, mouse_handlers, wp, style, erase_bg=not fl.transparent(), z_index=z_index)
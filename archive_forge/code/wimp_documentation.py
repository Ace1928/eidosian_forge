from __future__ import annotations
import typing
from urwid.canvas import CompositeCanvas
from urwid.command_map import Command
from urwid.signals import connect_signal
from urwid.text_layout import calc_coords
from urwid.util import is_mouse_press
from .columns import Columns
from .constants import Align, WrapMode
from .text import Text
from .widget import WidgetError, WidgetWrap

        Send 'click' signal on button 1 press.

        >>> size = (15,)
        >>> b = Button(u"Ok")
        >>> clicked_buttons = []
        >>> def handle_click(button):
        ...     clicked_buttons.append(button.label)
        >>> key = connect_signal(b, 'click', handle_click)
        >>> b.mouse_event(size, 'mouse press', 1, 4, 0, True)
        True
        >>> b.mouse_event(size, 'mouse press', 2, 4, 0, True) # ignored
        False
        >>> clicked_buttons # ... = u in Python 2
        [...'Ok']
        
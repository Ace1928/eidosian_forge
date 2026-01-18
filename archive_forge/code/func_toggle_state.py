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
def toggle_state(self) -> None:
    """
        Set state to True.

        >>> bgroup = [] # button group
        >>> b1 = RadioButton(bgroup, "Agree")
        >>> b2 = RadioButton(bgroup, "Disagree")
        >>> b1.state, b2.state
        (True, False)
        >>> b2.toggle_state()
        >>> b1.state, b2.state
        (False, True)
        >>> b2.toggle_state()
        >>> b1.state, b2.state
        (False, True)
        """
    self.set_state(True)
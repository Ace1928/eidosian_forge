from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
def unhandled_input(self, size: tuple[int, int], data: str) -> str | None:
    """Handle macro-navigation keys"""
    if data == 'left':
        self.move_focus_to_parent(size)
        return None
    if data == '-':
        self.collapse_focus_parent(size)
        return None
    return data
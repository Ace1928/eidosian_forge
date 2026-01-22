from __future__ import annotations
import string
import typing
from urwid import text_layout
from urwid.canvas import CompositeCanvas
from urwid.command_map import Command
from urwid.split_repr import remove_defaults
from urwid.str_util import is_wide_char, move_next_char, move_prev_char
from urwid.util import decompose_tagmarkup
from .constants import Align, Sizing, WrapMode
from .text import Text, TextError
class IntEdit(Edit):
    """Edit widget for integer values"""

    def valid_char(self, ch: str) -> bool:
        """
        Return true for decimal digits.
        """
        return len(ch) == 1 and ch in string.digits

    def __init__(self, caption='', default: int | str | None=None) -> None:
        """
        caption -- caption markup
        default -- default edit value

        >>> IntEdit(u"", 42)
        <IntEdit selectable flow widget '42' edit_pos=2>
        """
        if default is not None:
            val = str(default)
        else:
            val = ''
        super().__init__(caption, val)

    def keypress(self, size: tuple[int], key: str) -> str | None:
        """
        Handle editing keystrokes.  Remove leading zeros.

        >>> e, size = IntEdit(u"", 5002), (10,)
        >>> e.keypress(size, 'home')
        >>> e.keypress(size, 'delete')
        >>> print(e.edit_text)
        002
        >>> e.keypress(size, 'end')
        >>> print(e.edit_text)
        2
        """
        unhandled = super().keypress(size, key)
        if not unhandled:
            while self.edit_pos > 0 and self.edit_text[:1] == '0':
                self.set_edit_pos(self.edit_pos - 1)
                self.set_edit_text(self.edit_text[1:])
        return unhandled

    def value(self) -> int:
        """
        Return the numeric value of self.edit_text.

        >>> e, size = IntEdit(), (10,)
        >>> e.keypress(size, '5')
        >>> e.keypress(size, '1')
        >>> e.value() == 51
        True
        """
        if self.edit_text:
            return int(self.edit_text)
        return 0
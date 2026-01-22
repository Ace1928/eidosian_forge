from __future__ import annotations
import re
import warnings
from decimal import Decimal
from typing import TYPE_CHECKING
from urwid import Edit
class NumEdit(Edit):
    """NumEdit - edit numerical types

    based on the characters in 'allowed' different numerical types
    can be edited:
      + regular int: 0123456789
      + regular float: 0123456789.
      + regular oct: 01234567
      + regular hex: 0123456789abcdef
    """
    ALLOWED = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __init__(self, allowed: Container[str], caption, default: str | bytes, trimLeadingZeros: bool | None=None, *, trim_leading_zeros: bool=True, allow_negative: bool=False):
        super().__init__(caption, default)
        self._allowed = allowed
        self._trim_leading_zeros = trim_leading_zeros
        self._allow_negative = allow_negative
        if trimLeadingZeros is not None:
            warnings.warn("'trimLeadingZeros' argument is deprecated. Use 'trim_leading_zeros' keyword argument", DeprecationWarning, stacklevel=3)
            self._trim_leading_zeros = trimLeadingZeros

    def valid_char(self, ch: str) -> bool:
        """
        Return true for allowed characters.
        """
        if len(ch) == 1:
            if ch.upper() in self._allowed:
                return True
            return self._allow_negative and ch == '-' and (self.edit_pos == 0) and ('-' not in self.edit_text)
        return False

    def keypress(self, size: tuple[int], key: str) -> str | None:
        """
        Handle editing keystrokes.  Remove leading zeros.

        >>> e, size = NumEdit("0123456789", "", "5002"), (10,)
        >>> e.keypress(size, 'home')
        >>> e.keypress(size, 'delete')
        >>> assert e.edit_text == "002"
        >>> e.keypress(size, 'end')
        >>> assert e.edit_text == "2"
        >>> # binary only
        >>> e, size = NumEdit("01", "", ""), (10,)
        >>> assert e.edit_text == ""
        >>> e.keypress(size, '1')
        >>> e.keypress(size, '0')
        >>> e.keypress(size, '1')
        >>> assert e.edit_text == "101"
        >>> e, size = NumEdit("0123456789", "", "", allow_negative=True), (10,)
        >>> e.keypress(size, "-")
        >>> e.keypress(size, '1')
        >>> e.edit_text
        '-1'
        >>> e.keypress(size, 'home')
        >>> e.keypress(size, 'delete')
        >>> e.edit_text
        '1'
        >>> e.keypress(size, 'end')
        >>> e.keypress(size, "-")
        '-'
        >>> e.edit_text
        '1'
        """
        unhandled = super().keypress(size, key)
        if not unhandled and self._trim_leading_zeros:
            while self.edit_pos > 0 and self.edit_text[:1] == '0':
                self.set_edit_pos(self.edit_pos - 1)
                self.set_edit_text(self.edit_text[1:])
        return unhandled
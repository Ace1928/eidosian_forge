from __future__ import annotations
import re
import warnings
from decimal import Decimal
from typing import TYPE_CHECKING
from urwid import Edit
class IntegerEdit(NumEdit):
    """Edit widget for integer values"""

    def __init__(self, caption='', default: int | str | Decimal | None=None, base: int=10, *, allow_negative: bool=False) -> None:
        """
        caption -- caption markup
        default -- default edit value

        >>> IntegerEdit(u"", 42)
        <IntegerEdit selectable flow widget '42' edit_pos=2>
        >>> e, size = IntegerEdit(u"", "5002"), (10,)
        >>> e.keypress(size, 'home')
        >>> e.keypress(size, 'delete')
        >>> assert e.edit_text == "002"
        >>> e.keypress(size, 'end')
        >>> assert e.edit_text == "2"
        >>> e.keypress(size, '9')
        >>> e.keypress(size, '0')
        >>> assert e.edit_text == "290"
        >>> e, size = IntegerEdit("", ""), (10,)
        >>> assert e.value() is None
        >>> # binary
        >>> e, size = IntegerEdit(u"", "1010", base=2), (10,)
        >>> e.keypress(size, 'end')
        >>> e.keypress(size, '1')
        >>> assert e.edit_text == "10101"
        >>> assert e.value() == Decimal("21")
        >>> # HEX
        >>> e, size = IntegerEdit(u"", "10", base=16), (10,)
        >>> e.keypress(size, 'end')
        >>> e.keypress(size, 'F')
        >>> e.keypress(size, 'f')
        >>> assert e.edit_text == "10Ff"
        >>> assert e.keypress(size, 'G') == 'G'  # unhandled key
        >>> assert e.edit_text == "10Ff"
        >>> # keep leading 0's when not base 10
        >>> e, size = IntegerEdit(u"", "10FF", base=16), (10,)
        >>> assert e.edit_text == "10FF"
        >>> assert e.value() == Decimal("4351")
        >>> e.keypress(size, 'home')
        >>> e.keypress(size, 'delete')
        >>> e.keypress(size, '0')
        >>> assert e.edit_text == "00FF"
        >>> # test exception on incompatible value for base
        >>> e, size = IntegerEdit(u"", "10FG", base=16), (10,)
        Traceback (most recent call last):
            ...
        ValueError: invalid value: 10FG for base 16
        >>> # test exception on float init value
        >>> e, size = IntegerEdit(u"", 10.0), (10,)
        Traceback (most recent call last):
            ...
        ValueError: default: Only 'str', 'int', 'long' or Decimal input allowed
        >>> e, size = IntegerEdit(u"", Decimal("10.0")), (10,)
        Traceback (most recent call last):
            ...
        ValueError: not an 'integer Decimal' instance
        """
        self.base = base
        val = ''
        allowed_chars = self.ALLOWED[:self.base]
        if default is not None:
            if not isinstance(default, (int, str, Decimal)):
                raise ValueError("default: Only 'str', 'int' or Decimal input allowed")
            if isinstance(default, str) and len(default):
                validation_re = f'^[{allowed_chars}]+$'
                if not re.match(validation_re, str(default), re.IGNORECASE):
                    raise ValueError(f'invalid value: {default} for base {base}')
            elif isinstance(default, Decimal) and default.as_tuple()[2] != 0:
                raise ValueError("not an 'integer Decimal' instance")
            val = str(default)
        super().__init__(allowed_chars, caption, val, trim_leading_zeros=self.base == 10, allow_negative=allow_negative)

    def value(self) -> Decimal | None:
        """
        Return the numeric value of self.edit_text.

        >>> e, size = IntegerEdit(), (10,)
        >>> e.keypress(size, '5')
        >>> e.keypress(size, '1')
        >>> assert e.value() == 51
        """
        if self.edit_text:
            return Decimal(int(self.edit_text, self.base))
        return None

    def __int__(self) -> int:
        """Enforced int value return.

        >>> e, size = IntegerEdit(allow_negative=True), (10,)
        >>> assert int(e) == 0
        >>> e.keypress(size, '-')
        >>> e.keypress(size, '4')
        >>> e.keypress(size, '2')
        >>> assert int(e) == -42
        """
        if self.edit_text:
            return int(self.edit_text, self.base)
        return 0
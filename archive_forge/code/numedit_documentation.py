from __future__ import annotations
import re
import warnings
from decimal import Decimal
from typing import TYPE_CHECKING
from urwid import Edit
Enforced float value return.

        >>> e, size = FloatEdit(allow_negative=True), (10,)
        >>> assert float(e) == 0.
        >>> e.keypress(size, '-')
        >>> e.keypress(size, '4')
        >>> e.keypress(size, '.')
        >>> e.keypress(size, '2')
        >>> assert float(e) == -4.2
        
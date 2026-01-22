from __future__ import annotations
import abc
import copy
import dataclasses
import math
import re
import string
import sys
from datetime import date
from datetime import datetime
from datetime import time
from datetime import tzinfo
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterable
from typing import Iterator
from typing import Sequence
from typing import TypeVar
from typing import cast
from typing import overload
from tomlkit._compat import PY38
from tomlkit._compat import decode
from tomlkit._types import _CustomDict
from tomlkit._types import _CustomFloat
from tomlkit._types import _CustomInt
from tomlkit._types import _CustomList
from tomlkit._types import wrap_method
from tomlkit._utils import CONTROL_CHARS
from tomlkit._utils import escape_string
from tomlkit.exceptions import InvalidStringError
class InlineTable(AbstractTable):
    """
    An inline table literal.
    """

    def __init__(self, value: container.Container, trivia: Trivia, new: bool=False) -> None:
        super().__init__(value, trivia)
        self._new = new

    @property
    def discriminant(self) -> int:
        return 10

    def append(self, key: Key | str | None, _item: Any) -> InlineTable:
        """
        Appends a (key, item) to the table.
        """
        if not isinstance(_item, Item):
            _item = item(_item, _parent=self)
        if not isinstance(_item, (Whitespace, Comment)):
            if not _item.trivia.indent and len(self._value) > 0 and (not self._new):
                _item.trivia.indent = ' '
            if _item.trivia.comment:
                _item.trivia.comment = ''
        self._value.append(key, _item)
        if isinstance(key, Key):
            key = key.key
        if key is not None:
            dict.__setitem__(self, key, _item)
        return self

    def as_string(self) -> str:
        buf = '{'
        last_item_idx = next((i for i in range(len(self._value.body) - 1, -1, -1) if self._value.body[i][0] is not None), None)
        for i, (k, v) in enumerate(self._value.body):
            if k is None:
                if i == len(self._value.body) - 1:
                    if self._new:
                        buf = buf.rstrip(', ')
                    else:
                        buf = buf.rstrip(',')
                buf += v.as_string()
                continue
            v_trivia_trail = v.trivia.trail.replace('\n', '')
            buf += f'{v.trivia.indent}{k.as_string() + ('.' if k.is_dotted() else '')}{k.sep}{v.as_string()}{v.trivia.comment}{v_trivia_trail}'
            if last_item_idx is not None and i < last_item_idx:
                buf += ','
                if self._new:
                    buf += ' '
        buf += '}'
        return buf

    def __setitem__(self, key: Key | str, value: Any) -> None:
        if hasattr(value, 'trivia') and value.trivia.comment:
            value.trivia.comment = ''
        super().__setitem__(key, value)

    def __copy__(self) -> InlineTable:
        return type(self)(self._value.copy(), self._trivia.copy(), self._new)

    def _getstate(self, protocol: int=3) -> tuple:
        return (self._value, self._trivia)
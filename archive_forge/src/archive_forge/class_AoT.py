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
class AoT(Item, _CustomList):
    """
    An array of table literal
    """

    def __init__(self, body: list[Table], name: str | None=None, parsed: bool=False) -> None:
        self.name = name
        self._body: list[Table] = []
        self._parsed = parsed
        super().__init__(Trivia(trail=''))
        for table in body:
            self.append(table)

    def unwrap(self) -> list[dict[str, Any]]:
        unwrapped = []
        for t in self._body:
            if hasattr(t, 'unwrap'):
                unwrapped.append(t.unwrap())
            else:
                unwrapped.append(t)
        return unwrapped

    @property
    def body(self) -> list[Table]:
        return self._body

    @property
    def discriminant(self) -> int:
        return 12

    @property
    def value(self) -> list[dict[Any, Any]]:
        return [v.value for v in self._body]

    def __len__(self) -> int:
        return len(self._body)

    @overload
    def __getitem__(self, key: slice) -> list[Table]:
        ...

    @overload
    def __getitem__(self, key: int) -> Table:
        ...

    def __getitem__(self, key):
        return self._body[key]

    def __setitem__(self, key: slice | int, value: Any) -> None:
        raise NotImplementedError

    def __delitem__(self, key: slice | int) -> None:
        del self._body[key]
        list.__delitem__(self, key)

    def insert(self, index: int, value: dict) -> None:
        value = item(value, _parent=self)
        if not isinstance(value, Table):
            raise ValueError(f'Unsupported insert value type: {type(value)}')
        length = len(self)
        if index < 0:
            index += length
        if index < 0:
            index = 0
        elif index >= length:
            index = length
        m = re.match('(?s)^[^ ]*([ ]+).*$', self._trivia.indent)
        if m:
            indent = m.group(1)
            m = re.match('(?s)^([^ ]*)(.*)$', value.trivia.indent)
            if not m:
                value.trivia.indent = indent
            else:
                value.trivia.indent = m.group(1) + indent + m.group(2)
        prev_table = self._body[index - 1] if 0 < index and length else None
        next_table = self._body[index + 1] if index < length - 1 else None
        if not self._parsed:
            if prev_table and '\n' not in value.trivia.indent:
                value.trivia.indent = '\n' + value.trivia.indent
            if next_table and '\n' not in next_table.trivia.indent:
                next_table.trivia.indent = '\n' + next_table.trivia.indent
        self._body.insert(index, value)
        list.insert(self, index, value)

    def invalidate_display_name(self):
        """Call ``invalidate_display_name`` on the contained tables"""
        for child in self:
            if hasattr(child, 'invalidate_display_name'):
                child.invalidate_display_name()

    def as_string(self) -> str:
        b = ''
        for table in self._body:
            b += table.as_string()
        return b

    def __repr__(self) -> str:
        return f'<AoT {self.value}>'

    def _getstate(self, protocol=3):
        return (self._body, self.name, self._parsed)
from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
class _NormalizedString:

    def __init__(self, *args: str) -> None:
        self._strs: list[str] = []
        for arg in args:
            self.append(arg)

    def append(self, s: str) -> None:
        self._strs.append(s.strip())

    def denormalize(self) -> str:
        return ''.join(map(unescape, self._strs))

    def __bool__(self) -> bool:
        return bool(self._strs)

    def __repr__(self) -> str:
        return os.linesep.join(self._strs)

    def __cmp__(self, other: object) -> int:
        if not other:
            return 1
        return _cmp(str(self), str(other))

    def __gt__(self, other: object) -> bool:
        return self.__cmp__(other) > 0

    def __lt__(self, other: object) -> bool:
        return self.__cmp__(other) < 0

    def __ge__(self, other: object) -> bool:
        return self.__cmp__(other) >= 0

    def __le__(self, other: object) -> bool:
        return self.__cmp__(other) <= 0

    def __eq__(self, other: object) -> bool:
        return self.__cmp__(other) == 0

    def __ne__(self, other: object) -> bool:
        return self.__cmp__(other) != 0
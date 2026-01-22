from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def add_pending(self, key: Key, flag: int) -> None:
    self._pending_flags.add((key, flag))
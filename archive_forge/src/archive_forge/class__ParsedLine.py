from __future__ import annotations
from .exceptions import ParseError
from typing import NamedTuple
class _ParsedLine(NamedTuple):
    lineno: int
    section: str | None
    name: str | None
    value: str | None
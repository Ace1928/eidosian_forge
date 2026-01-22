from __future__ import annotations
from typing import NamedTuple
class _TagInfo(NamedTuple):
    value: int | None
    name: str
    type: int | None
    length: int | None
    enum: dict[str, int]
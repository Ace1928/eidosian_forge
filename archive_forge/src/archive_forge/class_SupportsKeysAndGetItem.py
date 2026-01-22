from __future__ import annotations
from typing import (
class SupportsKeysAndGetItem(Protocol):
    """
    Dict-like types with ``keys() -> str`` and ``__getitem__(key: str) -> str`` methods.

    """

    def keys(self) -> Iterable[str]:
        ...

    def __getitem__(self, key: str) -> str:
        ...
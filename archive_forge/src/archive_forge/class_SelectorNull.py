from __future__ import annotations
import copyreg
from .pretty import pretty
from typing import Any, Iterator, Hashable, Pattern, Iterable, Mapping
class SelectorNull(Immutable):
    """Null Selector."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
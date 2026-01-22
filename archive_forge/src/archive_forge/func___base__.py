from __future__ import annotations
import copyreg
from .pretty import pretty
from typing import Any, Iterator, Hashable, Pattern, Iterable, Mapping
@classmethod
def __base__(cls) -> type[Immutable]:
    """Get base class."""
    return cls
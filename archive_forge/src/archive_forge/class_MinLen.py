import math
import sys
from dataclasses import dataclass
from datetime import timezone
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, SupportsFloat, SupportsIndex, TypeVar, Union
@dataclass(frozen=True, **SLOTS)
class MinLen(BaseMetadata):
    """
    MinLen() implies minimum inclusive length,
    e.g. ``len(value) >= min_length``.
    """
    min_length: Annotated[int, Ge(0)]
import math
import sys
from dataclasses import dataclass
from datetime import timezone
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, SupportsFloat, SupportsIndex, TypeVar, Union
class SupportsMod(Protocol):

    def __mod__(self: T, __other: T) -> T:
        ...
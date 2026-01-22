from typing import Any, Iterator, Optional, Sequence
from ..utils.base64 import base64, unbase64
from .connection import (
class SizedSliceable(Protocol):

    def __getitem__(self, index: slice) -> Any:
        ...

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...
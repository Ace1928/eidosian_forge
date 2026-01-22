from __future__ import annotations
from collections.abc import (
from datetime import (
from os import PathLike
import sys
from typing import (
import numpy as np
class BaseBuffer(Protocol):

    @property
    def mode(self) -> str:
        ...

    def seek(self, __offset: int, __whence: int=...) -> int:
        ...

    def seekable(self) -> bool:
        ...

    def tell(self) -> int:
        ...
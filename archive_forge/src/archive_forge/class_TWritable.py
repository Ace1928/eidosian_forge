from __future__ import annotations
import os
import pathlib
from types import FrameType, ModuleType
from typing import (
class TWritable(Protocol):
    """Anything that can be written to."""

    def write(self, msg: str) -> None:
        """Write a message."""
from __future__ import annotations
from collections.abc import (
from datetime import (
from os import PathLike
import sys
from typing import (
import numpy as np
class WriteExcelBuffer(WriteBuffer[bytes], Protocol):

    def truncate(self, size: int | None=...) -> int:
        ...
from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
@property
def _header_row_num(self) -> int:
    """Number of rows in header."""
    return self.header_levels if self.fmt.header else 0
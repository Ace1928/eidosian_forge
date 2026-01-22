from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def _preprocess_row(self, row: Sequence[str]) -> list[str]:
    """Preprocess elements of the row."""
    if self.fmt.escape:
        crow = _escape_symbols(row)
    else:
        crow = [x if x else '{}' for x in row]
    if self.fmt.bold_rows and self.fmt.index:
        crow = _convert_to_bold(crow, self.index_levels)
    return crow
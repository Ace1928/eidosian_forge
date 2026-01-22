from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def _get_index_format(self) -> str:
    """Get index column format."""
    return 'l' * self.frame.index.nlevels if self.fmt.index else ''
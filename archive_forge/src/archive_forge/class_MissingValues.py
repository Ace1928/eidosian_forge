from collections import Counter
from contextlib import suppress
from typing import NamedTuple
import numpy as np
from . import is_scalar_nan
class MissingValues(NamedTuple):
    """Data class for missing data information"""
    nan: bool
    none: bool

    def to_list(self):
        """Convert tuple to a list where None is always first."""
        output = []
        if self.none:
            output.append(None)
        if self.nan:
            output.append(np.nan)
        return output
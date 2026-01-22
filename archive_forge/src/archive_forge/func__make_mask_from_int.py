from __future__ import annotations
from collections.abc import Iterable
from typing import (
import numpy as np
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
def _make_mask_from_int(self, arg: int) -> np.ndarray:
    if arg >= 0:
        return self._ascending_count == arg
    else:
        return self._descending_count == -arg - 1
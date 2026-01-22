from __future__ import annotations
from collections.abc import Iterable
from typing import (
import numpy as np
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
@cache_readonly
def _descending_count(self) -> np.ndarray:
    if TYPE_CHECKING:
        groupby_self = cast(groupby.GroupBy, self)
    else:
        groupby_self = self
    return groupby_self._cumcount_array(ascending=False)
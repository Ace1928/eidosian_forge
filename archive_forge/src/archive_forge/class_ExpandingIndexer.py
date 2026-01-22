from __future__ import annotations
from datetime import timedelta
import numpy as np
from pandas._libs.tslibs import BaseOffset
from pandas._libs.window.indexers import calculate_variable_window_bounds
from pandas.util._decorators import Appender
from pandas.core.dtypes.common import ensure_platform_int
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.tseries.offsets import Nano
class ExpandingIndexer(BaseIndexer):
    """Calculate expanding window bounds, mimicking df.expanding()"""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values: int=0, min_periods: int | None=None, center: bool | None=None, closed: str | None=None, step: int | None=None) -> tuple[np.ndarray, np.ndarray]:
        return (np.zeros(num_values, dtype=np.int64), np.arange(1, num_values + 1, dtype=np.int64))
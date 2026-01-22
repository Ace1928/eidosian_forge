from __future__ import annotations
from datetime import timedelta
import numpy as np
from pandas._libs.tslibs import BaseOffset
from pandas._libs.window.indexers import calculate_variable_window_bounds
from pandas.util._decorators import Appender
from pandas.core.dtypes.common import ensure_platform_int
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.tseries.offsets import Nano
class VariableWindowIndexer(BaseIndexer):
    """Creates window boundaries that are of variable length, namely for time series."""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(self, num_values: int=0, min_periods: int | None=None, center: bool | None=None, closed: str | None=None, step: int | None=None) -> tuple[np.ndarray, np.ndarray]:
        return calculate_variable_window_bounds(num_values, self.window_size, min_periods, center, closed, self.index_array)
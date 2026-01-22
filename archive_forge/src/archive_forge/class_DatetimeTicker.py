from __future__ import annotations
import logging # isort:skip
from ..core.enums import LatLon
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from .mappers import ScanningColorMapper
class DatetimeTicker(CompositeTicker):
    """ Generate nice ticks across different date and time scales.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    num_minor_ticks = Override(default=0)
    tickers = Override(default=lambda: [AdaptiveTicker(mantissas=[1, 2, 5], base=10, min_interval=0, max_interval=500 * ONE_MILLI, num_minor_ticks=0), AdaptiveTicker(mantissas=[1, 2, 5, 10, 15, 20, 30], base=60, min_interval=ONE_SECOND, max_interval=30 * ONE_MINUTE, num_minor_ticks=0), AdaptiveTicker(mantissas=[1, 2, 4, 6, 8, 12], base=24, min_interval=ONE_HOUR, max_interval=12 * ONE_HOUR, num_minor_ticks=0), DaysTicker(days=list(range(1, 32))), DaysTicker(days=list(range(1, 31, 3))), DaysTicker(days=[1, 8, 15, 22]), DaysTicker(days=[1, 15]), MonthsTicker(months=list(range(0, 12, 1))), MonthsTicker(months=list(range(0, 12, 2))), MonthsTicker(months=list(range(0, 12, 4))), MonthsTicker(months=list(range(0, 12, 6))), YearsTicker()])
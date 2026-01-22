from __future__ import annotations
import typing
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
import dateutil.rrule as rr
@dataclass
class date_breaks_info:
    """
    Information required to generate sequence of date breaks
    """
    frequency: DateFrequency
    n: int
    width: int
    start: datetime
    until: datetime
    tz: TzInfo | None
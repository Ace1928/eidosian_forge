from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from isoduration.formatter import format_duration
from isoduration.operations import add
@dataclass
class TimeDuration:
    hours: Decimal = Decimal(0)
    minutes: Decimal = Decimal(0)
    seconds: Decimal = Decimal(0)

    def __neg__(self) -> TimeDuration:
        return TimeDuration(hours=-self.hours, minutes=-self.minutes, seconds=-self.seconds)
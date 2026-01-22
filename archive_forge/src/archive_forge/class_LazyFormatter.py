import time
import dateparser
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Callable
class LazyFormatter:

    def size(cls, bvalue, suffix='B'):
        factor = 1024
        for unit in ['', 'K', 'M', 'G', 'T', 'P']:
            if bvalue < factor:
                return LazyData(string=f'{bvalue:.2f} {unit}{suffix}', value=bvalue, dtype='size')
            bvalue /= factor

    def ftime(cls, t: time.time=None, short: bool=True):
        return LazyTime(t, short=short)

    def dtime(cls, dt: str=None, prefer: str='past'):
        return LazyDate.dtime(dt, prefer)
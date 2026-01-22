from abc import abstractmethod
import math
import operator
import re
import datetime
from calendar import isleap
from decimal import Decimal, Context
from typing import cast, Any, Callable, Dict, Optional, Tuple, Union
from ..helpers import MONTH_DAYS_LEAP, MONTH_DAYS, DAYS_IN_4Y, \
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
def _date_operator(self, op: Callable[[Any, Any], Any], other: object) -> Union['DayTimeDuration', 'OrderedDateTime']:
    if isinstance(other, self.__class__):
        dt1, dt2 = self._get_operands(other)
        if self._year is None and other._year is None:
            return DayTimeDuration.fromtimedelta(dt1 - dt2)
        return DayTimeDuration.fromtimedelta(self.todelta() - other.todelta())
    elif isinstance(other, datetime.timedelta):
        delta = op(self.todelta(), other)
        return type(self).fromdelta(delta, adjust_timezone=True)
    elif isinstance(other, DayTimeDuration):
        delta = op(self.todelta(), other.get_timedelta())
        tzinfo = cast(Optional[Timezone], self._dt.tzinfo)
        if tzinfo is None:
            return type(self).fromdelta(delta)
        value = type(self).fromdelta(delta + tzinfo.offset)
        value.tzinfo = tzinfo
        return value
    elif isinstance(other, YearMonthDuration):
        month = op(self._dt.month - 1, other.months) % 12 + 1
        year = self.year + op(self._dt.month - 1, other.months) // 12
        day = adjust_day(year, month, self._dt.day)
        if year > 0:
            dt = self._dt.replace(year=year, month=month, day=day)
        elif isleap(year):
            dt = self._dt.replace(year=4, month=month, day=day)
        else:
            dt = self._dt.replace(year=6, month=month, day=day)
        kwargs = {k: getattr(dt, k) for k in self.pattern.groupindex.keys()}
        if year <= 0:
            kwargs['year'] = year
        return type(self)(**kwargs)
    else:
        raise TypeError('wrong type %r for operand %r' % (type(other), other))
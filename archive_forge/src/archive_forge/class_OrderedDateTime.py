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
class OrderedDateTime(AbstractDateTime):

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @classmethod
    def fromdelta(cls, delta: datetime.timedelta, adjust_timezone: bool=False) -> 'OrderedDateTime':
        """
        Creates an XSD dateTime/date instance from a datetime.timedelta related to
        0001-01-01T00:00:00 CE. In case of a date the time part is not counted.

        :param delta: a datetime.timedelta instance.
        :param adjust_timezone: if `True` adjusts the timezone of Date objects         with eventually present hours and minutes.
        """
        try:
            dt = datetime.datetime(1, 1, 1) + delta
        except OverflowError:
            days = delta.days
            if days > 0:
                y400, days = divmod(days, DAYS_IN_400Y)
                y100, days = divmod(days, DAYS_IN_100Y)
                y4, days = divmod(days, DAYS_IN_4Y)
                y1, days = divmod(days, 365)
                year = y400 * 400 + y100 * 100 + y4 * 4 + y1 + 1
                if y1 == 4 or y100 == 4:
                    year -= 1
                    days = 365
                td = datetime.timedelta(days=days, seconds=delta.seconds, microseconds=delta.microseconds)
                dt = datetime.datetime(4 if isleap(year) else 6, 1, 1) + td
            elif days >= -366:
                year = -1
                td = datetime.timedelta(days=days, seconds=delta.seconds, microseconds=delta.microseconds)
                dt = datetime.datetime(5, 1, 1) + td
            else:
                days = -days - 366
                y400, days = divmod(days, DAYS_IN_400Y)
                y100, days = divmod(days, DAYS_IN_100Y)
                y4, days = divmod(days, DAYS_IN_4Y)
                y1, days = divmod(days, 365)
                year = -y400 * 400 - y100 * 100 - y4 * 4 - y1 - 2
                if y1 == 4 or y100 == 4:
                    year += 1
                    days = 365
                td = datetime.timedelta(days=-days, seconds=delta.seconds, microseconds=delta.microseconds)
                if not td:
                    dt = datetime.datetime(4 if isleap(year + 1) else 6, 1, 1)
                    year += 1
                else:
                    dt = datetime.datetime(5 if isleap(year + 1) else 7, 1, 1) + td
        else:
            year = dt.year
        if issubclass(cls, Date10):
            if adjust_timezone and (dt.hour or dt.minute):
                assert dt.tzinfo is None
                hour, minute = (dt.hour, dt.minute)
                if hour < 14 or (hour == 14 and minute == 0):
                    tz = Timezone(datetime.timedelta(hours=-hour, minutes=-minute))
                    dt = dt.replace(tzinfo=tz)
                else:
                    tz = Timezone(datetime.timedelta(hours=-dt.hour + 24, minutes=-minute))
                    dt = dt.replace(tzinfo=tz)
                    dt += datetime.timedelta(days=1)
            return cls(year, dt.month, dt.day, tzinfo=dt.tzinfo)
        return cls(year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond, dt.tzinfo)

    def todelta(self) -> datetime.timedelta:
        """Returns the datetime.timedelta from 0001-01-01T00:00:00 CE."""
        if self._year is None:
            delta = operator.sub(*self._get_operands(datetime.datetime(1, 1, 1)))
            return cast(datetime.timedelta, delta)
        year, dt = (self.year, self._dt)
        tzinfo = None if dt.tzinfo is None else self._utc_timezone
        if year > 0:
            m_days = MONTH_DAYS_LEAP if isleap(year) else MONTH_DAYS
            days = days_from_common_era(year - 1) + sum((m_days[m] for m in range(1, dt.month)))
        else:
            m_days = MONTH_DAYS_LEAP if isleap(year + 1) else MONTH_DAYS
            days = days_from_common_era(year) + sum((m_days[m] for m in range(1, dt.month)))
        delta = dt - datetime.datetime(dt.year, dt.month, day=1, tzinfo=tzinfo)
        return datetime.timedelta(days=days, seconds=delta.total_seconds())

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

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, (AbstractDateTime, datetime.datetime)):
            return NotImplemented
        dt1, dt2 = self._get_operands(other)
        y1, y2 = (self.year, other.year)
        return y1 < y2 or (y1 == y2 and dt1 < dt2)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, (AbstractDateTime, datetime.datetime)):
            return NotImplemented
        dt1, dt2 = self._get_operands(other)
        y1, y2 = (self.year, other.year)
        return y1 < y2 or (y1 == y2 and dt1 <= dt2)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, (AbstractDateTime, datetime.datetime)):
            return NotImplemented
        dt1, dt2 = self._get_operands(other)
        y1, y2 = (self.year, other.year)
        return y1 > y2 or (y1 == y2 and dt1 > dt2)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, (AbstractDateTime, datetime.datetime)):
            return NotImplemented
        dt1, dt2 = self._get_operands(other)
        y1, y2 = (self.year, other.year)
        return y1 > y2 or (y1 == y2 and dt1 >= dt2)

    def __add__(self, other: object) -> Union['DayTimeDuration', 'OrderedDateTime']:
        if isinstance(other, OrderedDateTime):
            raise TypeError('wrong type %r for operand %r' % (type(other), other))
        return self._date_operator(operator.add, other)

    def __sub__(self, other: object) -> Union['DayTimeDuration', 'OrderedDateTime']:
        return self._date_operator(operator.sub, other)
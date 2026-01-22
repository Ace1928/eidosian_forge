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
class DayTimeDuration(Duration):
    name = 'dayTimeDuration'

    def __init__(self, seconds: Union[Decimal, int]=0) -> None:
        super(DayTimeDuration, self).__init__(0, seconds)

    @classmethod
    def fromtimedelta(cls, td: datetime.timedelta) -> 'DayTimeDuration':
        return cls(seconds=Decimal('{}.{:06}'.format(td.days * 86400 + td.seconds, td.microseconds)))

    def get_timedelta(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=int(self.seconds), microseconds=int(self.seconds % 1 * 1000000))

    def __repr__(self) -> str:
        return '%s(seconds=%s)' % (self.__class__.__name__, normalized_seconds(self.seconds))

    def __add__(self, other: object) -> Union['DayTimeDuration', Time, OrderedDateTime]:
        if isinstance(other, (Time, Date10)):
            return other + self
        elif isinstance(other, self.__class__):
            return DayTimeDuration(self.seconds + other.seconds)
        raise TypeError('cannot add %r to %r' % (type(other), type(self)))

    def __sub__(self, other: object) -> 'DayTimeDuration':
        if not isinstance(other, self.__class__):
            raise TypeError('cannot subtract %r from %r' % (type(other), type(self)))
        return DayTimeDuration(seconds=self.seconds - other.seconds)

    def __mul__(self, other: object) -> 'DayTimeDuration':
        if isinstance(other, (float, int, Decimal)):
            if math.isnan(other):
                raise ValueError('cannot multiply a %r by NaN' % type(self))
            if isinstance(other, (int, Decimal)):
                seconds = self.seconds * other
            else:
                seconds = self.seconds * Decimal.from_float(other)
            return DayTimeDuration(seconds)
        else:
            raise TypeError('cannot multiply a %r by %r' % (type(self), type(other)))

    def __truediv__(self, other: object) -> Union[Decimal, 'DayTimeDuration']:
        if isinstance(other, self.__class__):
            return self.seconds / other.seconds
        elif isinstance(other, (float, int, Decimal)):
            if math.isnan(other):
                raise ValueError('cannot divide a %r by NaN' % type(self))
            if isinstance(other, (int, Decimal)):
                seconds = self.seconds / other
            else:
                seconds = self.seconds / Decimal.from_float(other)
            return DayTimeDuration(seconds)
        else:
            raise TypeError('cannot divide a %r by %r' % (type(self), type(other)))
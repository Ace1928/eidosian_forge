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
class Timezone(datetime.tzinfo):
    """
    A tzinfo implementation for XSD timezone offsets. Offsets must be specified
    between -14:00 and +14:00.

    :param offset: a timedelta instance or an XSD timezone formatted string.
    """
    _maxoffset = datetime.timedelta(hours=14, minutes=0)
    _minoffset = -_maxoffset

    def __init__(self, offset: datetime.timedelta) -> None:
        super(Timezone, self).__init__()
        if not isinstance(offset, datetime.timedelta):
            raise TypeError('offset must be a datetime.timedelta')
        if offset < self._minoffset or offset > self._maxoffset:
            raise ValueError('offset must be between -14:00 and +14:00')
        self.offset = offset

    @classmethod
    def fromstring(cls, text: str) -> 'Timezone':
        try:
            hours, minutes = text.strip().split(':')
            if hours.startswith('-'):
                return cls(datetime.timedelta(hours=int(hours), minutes=-int(minutes)))
            else:
                return cls(datetime.timedelta(hours=int(hours), minutes=int(minutes)))
        except AttributeError:
            raise TypeError('argument is not a string')
        except ValueError:
            if text.strip() == 'Z':
                return cls(datetime.timedelta(0))
            raise ValueError('%r: not an XSD timezone formatted string' % text) from None

    @classmethod
    def fromduration(cls, duration: 'Duration') -> 'Timezone':
        if duration.seconds % 60 != 0:
            raise ValueError('{!r} has not an integral number of minutes'.format(duration))
        return cls(datetime.timedelta(seconds=int(duration.seconds)))

    def __getinitargs__(self) -> Tuple[datetime.timedelta]:
        return (self.offset,)

    def __hash__(self) -> int:
        return hash(self.offset)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Timezone) and self.offset == other.offset

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, Timezone) or self.offset != other.offset

    def __repr__(self) -> str:
        return '%s(%r)' % (self.__class__.__name__, self.offset)

    def __str__(self) -> str:
        return self.tzname(None)

    def utcoffset(self, dt: Optional[datetime.datetime]) -> datetime.timedelta:
        if not isinstance(dt, datetime.datetime) and dt is not None:
            raise TypeError('utcoffset() argument must be a datetime.datetime instance or None')
        return self.offset

    def tzname(self, dt: Optional[datetime.datetime]) -> str:
        if not isinstance(dt, datetime.datetime) and dt is not None:
            raise TypeError('tzname() argument must be a datetime.datetime instance or None')
        if not self.offset:
            return 'Z'
        elif self.offset < datetime.timedelta(0):
            sign, offset = ('-', -self.offset)
        else:
            sign, offset = ('+', self.offset)
        hours, minutes = (offset.seconds // 3600, offset.seconds // 60 % 60)
        return '{}{:02d}:{:02d}'.format(sign, hours, minutes)

    def dst(self, dt: Optional[datetime.datetime]) -> None:
        if not isinstance(dt, datetime.datetime) and dt is not None:
            raise TypeError('dst() argument must be a datetime.datetime instance or None')

    def fromutc(self, dt: datetime.datetime) -> datetime.datetime:
        if isinstance(dt, datetime.datetime):
            return dt + self.offset
        raise TypeError('fromutc() argument must be a datetime.datetime instance')
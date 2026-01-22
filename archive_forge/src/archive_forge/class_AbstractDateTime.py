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
class AbstractDateTime(AnyAtomicType):
    """
    A class for representing XSD date/time objects. It uses and internal datetime.datetime
    attribute and an integer attribute for processing BCE years or for years after 9999 CE.
    """
    xsd_version = '1.0'
    pattern = re.compile('^$')
    _utc_timezone = Timezone(datetime.timedelta(0))
    _year = None

    def __init__(self, year: int=2000, month: int=1, day: int=1, hour: int=0, minute: int=0, second: int=0, microsecond: int=0, tzinfo: Optional[datetime.tzinfo]=None) -> None:
        if hour == 24 and minute == second == microsecond == 0:
            delta = datetime.timedelta(days=1)
            hour = 0
        else:
            delta = datetime.timedelta(0)
        if 1 <= year <= 9999:
            self._dt = datetime.datetime(year, month, day, hour, minute, second, microsecond, tzinfo)
        elif year == 0:
            raise ValueError('0 is an illegal value for year')
        elif not isinstance(year, int):
            raise TypeError('invalid type %r for year' % type(year))
        elif abs(year) > 2 ** 31:
            raise OverflowError('year overflow')
        else:
            self._year = year
            if isleap(year + bool(self.xsd_version != '1.0')):
                self._dt = datetime.datetime(4, month, day, hour, minute, second, microsecond, tzinfo)
            else:
                self._dt = datetime.datetime(6, month, day, hour, minute, second, microsecond, tzinfo)
        if delta:
            self._dt += delta

    def __repr__(self) -> str:
        fields = self.pattern.groupindex.keys()
        arg_string = ', '.join((str(getattr(self, k)) for k in ['year', 'month', 'day', 'hour', 'minute'] if k in fields))
        if 'second' in fields:
            if self.microsecond:
                arg_string += ', %d.%06d' % (self.second, self.microsecond)
            else:
                arg_string += ', %d' % self.second
        if self.tzinfo is not None:
            arg_string += ', tzinfo=%r' % self.tzinfo
        return '%s(%s)' % (self.__class__.__name__, arg_string)

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def __lt__(self, other: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __le__(self, other: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __gt__(self, other: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __ge__(self, other: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other: object) -> Any:
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other: object) -> Any:
        raise NotImplementedError

    @property
    def year(self) -> int:
        return self._year or self._dt.year

    @property
    def bce(self) -> bool:
        return self._year is not None and self._year < 0

    @property
    def iso_year(self) -> str:
        """The ISO string representation of the year field."""
        year = self.year
        if -9999 <= year < -1:
            return '{:05}'.format(year if self.xsd_version == '1.0' else year + 1)
        elif year == -1:
            return '-0001' if self.xsd_version == '1.0' else '0000'
        elif 0 <= year <= 9999:
            return '{:04}'.format(year)
        else:
            return str(year)

    @property
    def month(self) -> int:
        return self._dt.month

    @property
    def day(self) -> int:
        return self._dt.day

    @property
    def hour(self) -> int:
        return self._dt.hour

    @property
    def minute(self) -> int:
        return self._dt.minute

    @property
    def second(self) -> int:
        return self._dt.second

    @property
    def microsecond(self) -> int:
        return self._dt.microsecond

    @property
    def tzinfo(self) -> Optional[Timezone]:
        return cast(Timezone, self._dt.tzinfo)

    @tzinfo.setter
    def tzinfo(self, tz: Timezone) -> None:
        self._dt = self._dt.replace(tzinfo=tz)

    def tzname(self) -> Optional[str]:
        return self._dt.tzname()

    def astimezone(self, tz: Optional[datetime.tzinfo]=None) -> datetime.datetime:
        return self._dt.astimezone(tz)

    def isocalendar(self) -> Tuple[int, int, int]:
        return self._dt.isocalendar()

    @classmethod
    def fromstring(cls, datetime_string: str, tzinfo: Optional[Timezone]=None) -> 'AbstractDateTime':
        """
        Creates an XSD date/time instance from a string formatted value.

        :param datetime_string: a string containing an XSD formatted date/time specification.
        :param tzinfo: optional implicit timezone information, must be a `Timezone` instance.
        :return: an AbstractDateTime concrete subclass instance.
        """
        if not isinstance(datetime_string, str):
            msg = '1st argument has an invalid type {!r}'
            raise TypeError(msg.format(type(datetime_string)))
        elif tzinfo and (not isinstance(tzinfo, Timezone)):
            msg = '2nd argument has an invalid type {!r}'
            raise TypeError(msg.format(type(tzinfo)))
        match = cls.pattern.match(datetime_string.strip())
        if match is None:
            msg = 'Invalid datetime string {!r} for {!r}'
            raise ValueError(msg.format(datetime_string, cls))
        match_dict = match.groupdict()
        kwargs: Dict[str, int] = {k: int(v) for k, v in match_dict.items() if k != 'tzinfo' and v is not None}
        if match_dict['tzinfo'] is not None:
            tzinfo = Timezone.fromstring(match_dict['tzinfo'])
        if 'microsecond' in kwargs:
            microseconds = match_dict['microsecond']
            if len(microseconds) != 6:
                microseconds += '0' * (6 - len(microseconds))
                kwargs['microsecond'] = int(microseconds[:6])
        if 'year' in kwargs:
            year_digits = match_dict['year'].lstrip('-')
            if year_digits.startswith('0') and len(year_digits) > 4:
                msg = 'Invalid datetime string {!r} for {!r} (when year exceeds 4 digits leading zeroes are not allowed)'
                raise ValueError(msg.format(datetime_string, cls))
            if cls.xsd_version == '1.0':
                if kwargs['year'] == 0:
                    raise ValueError("year '0000' is an illegal value for XSD 1.0")
            elif kwargs['year'] <= 0:
                kwargs['year'] -= 1
        return cls(tzinfo=tzinfo, **kwargs)

    @classmethod
    def fromdatetime(cls, dt: Union[datetime.datetime, datetime.date, datetime.time], year: Optional[int]=None) -> 'AbstractDateTime':
        """
        Creates an XSD date/time instance from a datetime.datetime/date/time instance.

        :param dt: the datetime, date or time instance that stores the XSD Date/Time value.
        :param year: if an year is provided the created instance refers to it and the         possibly present *dt.year* part is ignored.
        :return: an AbstractDateTime concrete subclass instance.
        """
        if not isinstance(dt, (datetime.datetime, datetime.date, datetime.time)):
            raise TypeError('1st argument has an invalid type %r' % type(dt))
        elif year is not None and (not isinstance(year, int)):
            raise TypeError('2nd argument has an invalid type %r' % type(year))
        kwargs = {k: getattr(dt, k) for k in cls.pattern.groupindex.keys() if hasattr(dt, k)}
        if year is not None:
            kwargs['year'] = year
        return cls(**kwargs)

    def _get_operands(self, other: object) -> Tuple[datetime.datetime, datetime.datetime]:
        if isinstance(other, (self.__class__, datetime.datetime)) or isinstance(self, other.__class__):
            dt: datetime.datetime = getattr(other, '_dt', cast(datetime.datetime, other))
            if self._dt.tzinfo is dt.tzinfo:
                return (self._dt, dt)
            elif self.tzinfo is None:
                return (self._dt.replace(tzinfo=self._utc_timezone), dt)
            elif dt.tzinfo is None:
                return (self._dt, dt.replace(tzinfo=self._utc_timezone))
            else:
                return (self._dt, dt)
        else:
            raise TypeError('wrong type %r for operand %r' % (type(other), other))

    def __hash__(self) -> int:
        return hash((self._dt, self._year))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (AbstractDateTime, datetime.datetime)):
            return False
        try:
            return operator.eq(*self._get_operands(other)) and self.year == other.year
        except TypeError:
            return False

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, (AbstractDateTime, datetime.datetime)):
            return True
        try:
            return operator.ne(*self._get_operands(other)) or self.year != other.year
        except TypeError:
            return True
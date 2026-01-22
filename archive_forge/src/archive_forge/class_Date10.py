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
class Date10(OrderedDateTime):
    """XSD 1.0 xs:date builtin type"""
    name = 'date'
    pattern = re.compile('^(?P<year>-?[0-9]*[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})(?P<tzinfo>Z|[+-](?:(?:0[0-9]|1[0-3]):[0-5][0-9]|14:00))?$')

    def __init__(self, year: int, month: int, day: int, tzinfo: Optional[datetime.tzinfo]=None) -> None:
        super(Date10, self).__init__(year, month, day, tzinfo=tzinfo)

    def __str__(self) -> str:
        return '{}-{:02}-{:02}{}'.format(self.iso_year, self.month, self.day, str(self.tzinfo or ''))
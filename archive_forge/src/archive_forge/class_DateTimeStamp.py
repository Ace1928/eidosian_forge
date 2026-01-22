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
class DateTimeStamp(DateTime):
    """XSD 1.1 xs:dateTimeStamp builtin type"""
    name = 'dateTimeStamp'
    pattern = re.compile('^(?P<year>-?[0-9]*[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})(T(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})(?:\\.(?P<microsecond>[0-9]+))?)(?P<tzinfo>Z|[+-](?:(?:0[0-9]|1[0-3]):[0-5][0-9]|14:00))$')
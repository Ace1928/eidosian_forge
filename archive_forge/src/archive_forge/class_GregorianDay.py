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
class GregorianDay(OrderedDateTime):
    """XSD xs:gDay builtin type"""
    name = 'gDay'
    pattern = re.compile('^---(?P<day>[0-9]{2})(?P<tzinfo>Z|[+-](?:(?:0[0-9]|1[0-3]):[0-5][0-9]|14:00))?$')

    def __init__(self, day: int, tzinfo: Optional[Timezone]=None) -> None:
        super(GregorianDay, self).__init__(day=day, tzinfo=tzinfo)

    def __str__(self) -> str:
        return '---{:02}{}'.format(self.day, str(self.tzinfo or ''))
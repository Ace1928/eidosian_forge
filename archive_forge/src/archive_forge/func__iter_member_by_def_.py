import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
@classmethod
def _iter_member_by_def_(cls, value):
    """
        Extract all members from the value in definition order.
        """
    yield from sorted(cls._iter_member_by_value_(value), key=lambda m: m._sort_order_)
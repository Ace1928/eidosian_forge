import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
def _power_of_two(value):
    if value < 1:
        return False
    return value == 2 ** _high_bit(value)
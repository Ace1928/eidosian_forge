import sys
import types
from array import array
from collections import abc
from ._abc import MultiMapping, MutableMultiMapping
class istr(str):
    """Case insensitive str."""
    __is_istr__ = True
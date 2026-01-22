import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
class member(object):
    """
    Forces item to become an Enum member during class creation.
    """

    def __init__(self, value):
        self.value = value
import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
class auto:
    """
    Instances are replaced with an appropriate value in Enum class suites.
    """

    def __init__(self, value=_auto_null):
        self.value = value

    def __repr__(self):
        return 'auto(%r)' % self.value
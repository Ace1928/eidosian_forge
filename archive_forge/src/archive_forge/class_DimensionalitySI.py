from collections import OrderedDict
from .pyutil import defaultnamedtuple
class DimensionalitySI(defaultnamedtuple('DimensionalitySIBase', dimension_codes.keys(), (0,) * len(dimension_codes))):

    def __mul__(self, other):
        return self.__class__(*(x + y for x, y in zip(self, other)))

    def __truediv__(self, other):
        return self.__class__(*(x - y for x, y in zip(self, other)))

    def __pow__(self, exp):
        return self.__class__(*(x * exp for x in self))
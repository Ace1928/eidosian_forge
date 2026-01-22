import subprocess
import sys
from math import nan, inf
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.environ import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
from pyomo.common.numeric_types import _native_boolean_types
class MyBogusNumericType(MyBogusType):

    def __add__(self, other):
        return MyBogusNumericType(self.val + float(other))

    def __lt__(self, other):
        return self.val < float(other)

    def __ge__(self, other):
        return self.val >= float(other)
import subprocess
import sys
from math import nan, inf
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.environ import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
from pyomo.common.numeric_types import _native_boolean_types
def test_param2(self):
    m = ConcreteModel()
    m.p = Param(mutable=True)
    self.assertEqual(0, polynomial_degree(m.p))
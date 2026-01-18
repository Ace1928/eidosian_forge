import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
def test_override_values(self):
    m = ConcreteModel()
    m.I = Set([1, 2, 3])
    m.I[1] = [1, 2, 3]
    self.assertEqual(sorted(m.I[1]), [1, 2, 3])
    m.I[1] = [4, 5, 6]
    self.assertEqual(sorted(m.I[1]), [4, 5, 6])
    m.J = Set([1, 2, 3], ordered=True)
    m.J[1] = [1, 3, 2]
    self.assertEqual(list(m.J[1]), [1, 3, 2])
    m.J[1] = [5, 4, 6]
    self.assertEqual(list(m.J[1]), [5, 4, 6])
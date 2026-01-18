import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.common.collections import ComponentSet
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set import (
from pyomo.core.base.indexed_component import UnindexedComponent_set, IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import (
def test_single_attribute_assignment(self):
    m = self.m
    rd = _ReferenceDict(m.b[1, 5].x[:, :])
    self.assertEqual(sum((x.value for x in rd.values())), 0)
    rd[7, 10].value = 10
    self.assertEqual(m.b[1, 5].x[7, 10].value, 10)
    self.assertEqual(sum((x.value for x in rd.values())), 10)
    rd = _ReferenceDict(m.b[1, 4].x[8, :])
    self.assertEqual(sum((x.value for x in rd.values())), 0)
    rd[10].value = 20
    self.assertEqual(m.b[1, 4].x[8, 10].value, 20)
    self.assertEqual(sum((x.value for x in rd.values())), 20)
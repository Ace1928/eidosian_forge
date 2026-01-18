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
def test_single_reference(self):
    m = ConcreteModel()
    m.b = Block([1, 2])
    m.b[1].x = Var(bounds=(1, None))
    m.b[2].x = Var(bounds=(2, None))
    m.r = Reference(m.b[:].x)
    self.assertIs(m.r.ctype, Var)
    self.assertIs(m.r.index_set(), m.b.index_set())
    self.assertEqual(len(m.r), 2)
    self.assertEqual(m.r[1].lb, 1)
    self.assertEqual(m.r[2].lb, 2)
    self.assertIn(1, m.r)
    self.assertIn(2, m.r)
    self.assertNotIn(3, m.r)
    with self.assertRaises(KeyError):
        m.r[3]
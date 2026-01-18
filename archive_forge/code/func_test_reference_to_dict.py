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
def test_reference_to_dict(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var([1, 2, 3])
    m.r = Reference({1: m.x, 'a': m.y[2], 3: m.y[1]})
    self.assertFalse(m.r.index_set().isordered())
    self.assertEqual(len(m.r), 3)
    self.assertEqual(set(m.r.keys()), {1, 3, 'a'})
    self.assertEqual(ComponentSet(m.r.values()), ComponentSet([m.x, m.y[2], m.y[1]]))
    del m.r[1]
    self.assertEqual(len(m.r), 2)
    self.assertEqual(set(m.r.keys()), {3, 'a'})
    self.assertEqual(ComponentSet(m.r.values()), ComponentSet([m.y[2], m.y[1]]))
    with self.assertRaisesRegex(KeyError, "Index '1' is not valid for indexed component 'r'"):
        m.r[1] = m.x
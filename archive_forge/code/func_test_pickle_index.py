import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_pickle_index(self):
    m = ConcreteModel()
    m.b = Block(Any)
    idx = "|b'foo'"
    m.b[idx].x = Var()
    cuid = ComponentUID(m.b[idx].x)
    self.assertEqual(str(cuid), 'b["|b\'foo\'"].x')
    self.assertIs(cuid.find_component_on(m), m.b[idx].x)
    tmp = ComponentUID(str(cuid))
    self.assertIsNot(cuid, tmp)
    self.assertEqual(cuid, tmp)
    self.assertIs(tmp.find_component_on(m), m.b[idx].x)
    tmp = pickle.loads(pickle.dumps(cuid))
    self.assertIsNot(cuid, tmp)
    self.assertEqual(cuid, tmp)
    self.assertIs(tmp.find_component_on(m), m.b[idx].x)
    idx = _Foo(1, 'a')
    m.b[idx].x = Var()
    cuid = ComponentUID(m.b[idx].x)
    self.assertRegex(str(cuid), '^b\\[\\|b?([\'\\"]).*\\.\\1\\]\\.x$')
    self.assertIs(cuid.find_component_on(m), m.b[idx].x)
    tmp = ComponentUID(str(cuid))
    self.assertIsNot(cuid, tmp)
    self.assertEqual(cuid, tmp)
    self.assertIs(tmp.find_component_on(m), m.b[idx].x)
    tmp = pickle.loads(pickle.dumps(cuid))
    self.assertIsNot(cuid, tmp)
    self.assertEqual(cuid, tmp)
    self.assertIs(tmp.find_component_on(m), m.b[idx].x)
    idx = datetime(1, 2, 3)
    m.b[idx].x = Var()
    cuid = ComponentUID(m.b[idx].x)
    self.assertRegex(str(cuid), '^b\\[\\|b?([\'\\"]).*\\.\\1\\]\\.x$')
    self.assertIs(cuid.find_component_on(m), m.b[idx].x)
    tmp = ComponentUID(str(cuid))
    self.assertIsNot(cuid, tmp)
    self.assertEqual(cuid, tmp)
    self.assertIs(tmp.find_component_on(m), m.b[idx].x)
    tmp = pickle.loads(pickle.dumps(cuid))
    self.assertIsNot(cuid, tmp)
    self.assertEqual(cuid, tmp)
    self.assertIs(tmp.find_component_on(m), m.b[idx].x)
    self.assertEqual(len(m.b), 3)
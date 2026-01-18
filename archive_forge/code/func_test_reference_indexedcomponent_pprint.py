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
def test_reference_indexedcomponent_pprint(self):
    m = ConcreteModel()
    m.x = Var([1, 2], initialize={1: 4, 2: 8})
    m.r = Reference(m.x, ctype=IndexedComponent)
    buf = StringIO()
    m.r.pprint(ostream=buf)
    self.assertEqual(buf.getvalue(), "r : Size=2, Index={1, 2}, ReferenceTo=x\n    Key : Object\n      1 : <class 'pyomo.core.base.var._GeneralVarData'>\n      2 : <class 'pyomo.core.base.var._GeneralVarData'>\n")
    m.s = Reference(m.x[:, ...], ctype=IndexedComponent)
    buf = StringIO()
    m.s.pprint(ostream=buf)
    self.assertEqual(buf.getvalue(), "s : Size=2, Index={1, 2}, ReferenceTo=x[:, ...]\n    Key : Object\n      1 : <class 'pyomo.core.base.var._GeneralVarData'>\n      2 : <class 'pyomo.core.base.var._GeneralVarData'>\n")
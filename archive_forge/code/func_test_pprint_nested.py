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
def test_pprint_nested(self):
    m = ConcreteModel()

    @m.Block([1, 2])
    def b(b, i):
        b.x = Var([3, 4], bounds=(i, None))
    m.r = Reference(m.b[:].x[:])
    buf = StringIO()
    m.r.pprint(ostream=buf)
    self.assertEqual(buf.getvalue().strip(), '\nr : Size=4, Index=ReferenceSet(b[:].x[:]), ReferenceTo=b[:].x[:]\n    Key    : Lower : Value : Upper : Fixed : Stale : Domain\n    (1, 3) :     1 :  None :  None : False :  True :  Reals\n    (1, 4) :     1 :  None :  None : False :  True :  Reals\n    (2, 3) :     2 :  None :  None : False :  True :  Reals\n    (2, 4) :     2 :  None :  None : False :  True :  Reals\n'.strip())
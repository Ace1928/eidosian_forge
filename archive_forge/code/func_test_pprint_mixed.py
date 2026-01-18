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
def test_pprint_mixed(self):
    m = ConcreteModel()
    m.Z = Set(initialize=['A', 'C'])
    m.A = Set(m.Z, initialize={'A': [1, 2, 3, 'A']})
    buf = StringIO()
    m.pprint(ostream=buf)
    ref = "2 Set Declarations\n    A : Size=1, Index=Z, Ordered=Insertion\n        Key : Dimen : Domain : Size : Members\n          A :     1 :    Any :    4 : {1, 2, 3, 'A'}\n    Z : Size=1, Index=None, Ordered=Insertion\n        Key  : Dimen : Domain : Size : Members\n        None :     1 :    Any :    2 : {'A', 'C'}\n\n2 Declarations: Z A\n"
    self.assertEqual(ref, buf.getvalue())
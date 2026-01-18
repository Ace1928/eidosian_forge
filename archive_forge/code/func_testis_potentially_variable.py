import pickle
import pyomo.common.unittest as unittest
from pyomo.core.expr.numvalue import (
import pyomo.kernel
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.expression import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.objective import objective
from pyomo.core.kernel.block import block
def testis_potentially_variable(self):
    e = self._ctype_factory()
    self.assertEqual(e.is_potentially_variable(), False)
    self.assertEqual(is_potentially_variable(e), False)
    e.expr = 1
    self.assertEqual(e.is_potentially_variable(), False)
    self.assertEqual(is_potentially_variable(e), False)
    p = parameter()
    e.expr = p ** 2
    self.assertEqual(e.is_potentially_variable(), False)
    self.assertEqual(is_potentially_variable(e), False)
    a = self._ctype_factory()
    e.expr = (a * p) ** 2 / (p + 5)
    self.assertEqual(e.is_potentially_variable(), False)
    self.assertEqual(is_potentially_variable(e), False)
    a.expr = 2.0
    p.value = 5.0
    self.assertEqual(e.is_potentially_variable(), False)
    self.assertEqual(is_potentially_variable(e), False)
    self.assertEqual(e(), 10.0)
    v = variable()
    with self.assertRaises(ValueError):
        e.expr = v + 1
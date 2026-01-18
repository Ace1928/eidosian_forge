import pickle
import pyomo.common.unittest as unittest
from pyomo.core.expr.numvalue import NumericValue
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.objective import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.block import block
def test_sense(self):
    o = objective()
    o.sense = maximize
    self.assertEqual(o.sense, maximize)
    self.assertEqual(o.is_minimizing(), False)
    with self.assertRaises(ValueError):
        o.sense = 100
    self.assertEqual(o.sense, maximize)
    self.assertEqual(o.is_minimizing(), False)
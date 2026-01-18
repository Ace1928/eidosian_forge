import pickle
import pyomo.common.unittest as unittest
from pyomo.core.expr import inequality, RangedExpression, EqualityExpression
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression, data_expression
from pyomo.core.kernel.block import block
def test_tuple_constraint_create(self):
    x = variable()
    y = variable()
    z = variable()
    c = constraint((0.0, x))
    with self.assertRaises(ValueError):
        constraint((y, x, z))
    with self.assertRaises(ValueError):
        constraint((0, x, z))
    with self.assertRaises(ValueError):
        constraint((y, x, 0))
    with self.assertRaises(ValueError):
        constraint((x, 0, 0, 0))
    c = constraint((x, y))
    self.assertEqual(c.upper, 0)
    self.assertEqual(c.lower, 0)
    self.assertTrue(c.body is not None)
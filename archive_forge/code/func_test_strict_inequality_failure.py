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
def test_strict_inequality_failure(self):
    x = variable()
    y = variable()
    c = constraint()
    with self.assertRaises(ValueError):
        c.expr = x < 0
    with self.assertRaises(ValueError):
        c.expr = inequality(body=x, upper=0, strict=True)
    c.expr = x <= 0
    c.expr = inequality(body=x, upper=0, strict=False)
    with self.assertRaises(ValueError):
        c.expr = x > 0
    with self.assertRaises(ValueError):
        c.expr = inequality(body=x, lower=0, strict=True)
    c.expr = x >= 0
    c.expr = inequality(body=x, lower=0, strict=False)
    with self.assertRaises(ValueError):
        c.expr = x < y
    with self.assertRaises(ValueError):
        c.expr = inequality(body=x, upper=y, strict=True)
    c.expr = x <= y
    c.expr = inequality(body=x, upper=y, strict=False)
    with self.assertRaises(ValueError):
        c.expr = x > y
    with self.assertRaises(ValueError):
        c.expr = inequality(body=x, lower=y, strict=True)
    c.expr = x >= y
    c.expr = inequality(body=x, lower=y, strict=False)
    with self.assertRaises(ValueError):
        c.expr = RangedExpression((0, x, 1), (True, True))
    with self.assertRaises(ValueError):
        c.expr = RangedExpression((0, x, 1), (False, True))
    with self.assertRaises(ValueError):
        c.expr = RangedExpression((0, x, 1), (True, False))
    c.expr = RangedExpression((0, x, 1), (False, False))
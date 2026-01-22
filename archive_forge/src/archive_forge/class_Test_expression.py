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
class Test_expression(_Test_expression_base, unittest.TestCase):
    _ctype_factory = expression

    def test_associativity(self):
        x = variable()
        y = variable()
        pyomo.kernel.pprint(y + x * expression(expression(x * y)))
        pyomo.kernel.pprint(y + expression(expression(x * y)) * x)

    def test_ctype(self):
        e = expression()
        self.assertIs(e.ctype, IExpression)
        self.assertIs(type(e), expression)
        self.assertIs(type(e)._ctype, IExpression)

    def test_is_fixed(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        e.expr = 1
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        v = variable()
        v.value = 2
        e.expr = v + 1
        self.assertEqual(e.is_fixed(), False)
        self.assertEqual(is_fixed(e), False)
        v.fix()
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(e(), 3)

    def testis_potentially_variable(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        e.expr = 1
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        v = variable()
        v.value = 2
        e.expr = v + 1
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        v.fix()
        e.expr = v + 1
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        self.assertEqual(e(), 3)

    def test_polynomial_degree(self):
        e = self._ctype_factory()
        e.expr = 1
        self.assertEqual(e.polynomial_degree(), 0)
        v = variable()
        v.value = 2
        e.expr = v + 1
        self.assertEqual(e.polynomial_degree(), 1)
        e.expr = v ** 2 + v + 1
        self.assertEqual(e.polynomial_degree(), 2)
        v.fix()
        self.assertEqual(e.polynomial_degree(), 0)
        e.expr = v ** v
        self.assertEqual(e.polynomial_degree(), 0)
        v.free()
        self.assertEqual(e.polynomial_degree(), None)
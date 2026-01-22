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
class Test_noclone(unittest.TestCase):

    def test_is_named_expression_type(self):
        e = expression()
        self.assertEqual(e.is_named_expression_type(), True)

    def test_arg(self):
        e = expression()
        self.assertEqual(e.arg(0), None)
        e.expr = 1
        self.assertEqual(e.arg(0), 1)
        with self.assertRaises(KeyError):
            e.arg(1)

    def test_init_non_NumericValue(self):
        types = [None, 1, 1.1, True, '']
        if numpy_available:
            types.extend([numpy.float32(1), numpy.bool_(True), numpy.int32(1)])
        types.append(block())
        types.append(block)
        for obj in types:
            self.assertEqual(noclone(obj), obj)
            self.assertIs(type(noclone(obj)), type(obj))

    def test_init_NumericValue(self):
        v = variable()
        p = parameter()
        e = expression()
        d = data_expression()
        o = objective()
        for obj in (v, v + 1, v ** 2, p, p + 1, p ** 2, e, e + 1, e ** 2, d, d + 1, d ** 2, o, o + 1, o ** 2):
            self.assertTrue(isinstance(noclone(obj), NumericValue))
            self.assertTrue(isinstance(noclone(obj), IIdentityExpression))
            self.assertIs(noclone(obj).expr, obj)

    def test_pprint(self):
        import pyomo.kernel
        v = variable()
        e = noclone(v ** 2)
        pyomo.kernel.pprint(e)
        pyomo.kernel.pprint(e, indent=1)
        b = block()
        b.e = expression(expr=e)
        pyomo.kernel.pprint(e)
        pyomo.kernel.pprint(b)
        m = block()
        m.b = b
        pyomo.kernel.pprint(e)
        pyomo.kernel.pprint(b)
        pyomo.kernel.pprint(m)
        pyomo.kernel.pprint(noclone(v) + 1)
        pyomo.kernel.pprint(noclone(v + 1))
        x = variable()
        y = variable()
        pyomo.kernel.pprint(y + x * noclone(noclone(x * y)))
        pyomo.kernel.pprint(y + noclone(noclone(x * y)) * x)

    def test_pickle(self):
        v = variable()
        e = noclone(v)
        self.assertEqual(type(e), expression)
        self.assertIs(type(e.expr), variable)
        eup = pickle.loads(pickle.dumps(e))
        self.assertEqual(type(eup), expression)
        self.assertTrue(e is not eup)
        self.assertIs(type(eup.expr), variable)
        self.assertIs(type(e.expr), variable)
        self.assertTrue(eup.expr is not e.expr)
        del e
        del v
        v = variable(value=1)
        b = block()
        b.v = v
        eraw = b.v + 1
        b.e = 1 + noclone(eraw)
        bup = pickle.loads(pickle.dumps(b))
        self.assertTrue(isinstance(bup.e, NumericValue))
        self.assertEqual(value(bup.e), 3.0)
        b.v.value = 2
        self.assertEqual(value(b.e), 4.0)
        self.assertEqual(value(bup.e), 3.0)
        bup.v.value = -1
        self.assertEqual(value(b.e), 4.0)
        self.assertEqual(value(bup.e), 1.0)
        self.assertIs(b.v.parent, b)
        self.assertIs(bup.v.parent, bup)
        del b.v

    def test_call(self):
        e = noclone(None)
        self.assertIs(e, None)
        e = noclone(1)
        self.assertEqual(e, 1)
        p = parameter()
        p.value = 2
        e = noclone(p + 1)
        self.assertEqual(e(), 3)

    def test_is_constant(self):
        v = variable()
        e = noclone(v)
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(is_constant(e), False)
        v.fix(1)
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(is_constant(e), False)
        p = parameter()
        e = noclone(p)
        self.assertEqual(p.is_constant(), False)
        self.assertEqual(is_constant(p), False)
        self.assertEqual(is_constant(noclone(1)), True)

    def test_is_fixed(self):
        v = variable()
        e = noclone(v + 1)
        self.assertEqual(e.is_fixed(), False)
        self.assertEqual(is_fixed(e), False)
        v.fix()
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)
        e = noclone(parameter())
        self.assertEqual(e.is_fixed(), True)
        self.assertEqual(is_fixed(e), True)

    def testis_potentially_variable(self):
        e = noclone(variable())
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        e = noclone(parameter())
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)
        e = noclone(expression())
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        e = noclone(data_expression())
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(is_potentially_variable(e), False)

    def test_polynomial_degree(self):
        e = noclone(parameter())
        self.assertEqual(e.polynomial_degree(), 0)
        e = noclone(parameter(value=1))
        self.assertEqual(e.polynomial_degree(), 0)
        v = variable()
        v.value = 2
        e = noclone(v + 1)
        self.assertEqual(e.polynomial_degree(), 1)
        e = noclone(v ** 2 + v + 1)
        self.assertEqual(e.polynomial_degree(), 2)
        v.fix()
        self.assertEqual(e.polynomial_degree(), 0)
        e = noclone(v ** v)
        self.assertEqual(e.polynomial_degree(), 0)
        v.free()
        self.assertEqual(e.polynomial_degree(), None)

    def test_is_expression_type(self):
        for obj in (variable(), parameter(), objective(), expression(), data_expression()):
            self.assertEqual(noclone(obj).is_expression_type(), True)

    def test_is_parameter_type(self):
        for obj in (variable(), parameter(), objective(), expression(), data_expression()):
            self.assertEqual(noclone(obj).is_parameter_type(), False)

    def test_args(self):
        e = noclone(parameter() + 1)
        self.assertEqual(e.nargs(), 1)
        self.assertTrue(e.arg(0) is e.expr)

    def test_arguments(self):
        e = noclone(parameter() + 1)
        self.assertEqual(len(tuple(e.args)), 1)
        self.assertTrue(tuple(e.args)[0] is e.expr)

    def test_clone(self):
        p = parameter()
        e = noclone(p)
        self.assertTrue(e.clone() is e)
        self.assertTrue(e.clone().expr is p)
        sube = p ** 2 + 1
        e = noclone(sube)
        self.assertTrue(e.clone() is e)
        self.assertTrue(e.clone().expr is sube)

    def test_division_behavior(self):
        e = noclone(parameter(value=2))
        self.assertIs(type(e.expr), parameter)
        self.assertEqual((1 / e)(), 0.5)
        self.assertEqual((parameter(1) / e)(), 0.5)
        self.assertEqual(1 / e.expr(), 0.5)

    def test_to_string(self):
        b = block()
        p = parameter()
        e = noclone(p ** 2)
        self.assertEqual(str(e.expr), '<parameter>**2')
        self.assertEqual(str(e), '<data_expression>')
        self.assertEqual(e.to_string(), '(<parameter>**2)')
        self.assertEqual(e.to_string(verbose=False), '(<parameter>**2)')
        self.assertEqual(e.to_string(verbose=True), '<data_expression>{pow(<parameter>, 2)}')
        b.e = e
        b.p = p
        self.assertNotEqual(p.name, None)
        self.assertEqual(e.to_string(verbose=True), 'e{pow(' + p.name + ', 2)}')
        self.assertEqual(e.to_string(verbose=True), 'e{pow(p, 2)}')
        del b.e
        del b.p
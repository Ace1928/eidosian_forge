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
class _Test_expression_base(object):
    _ctype_factory = None

    def test_pprint(self):
        import pyomo.kernel
        p = parameter()
        e = self._ctype_factory(p ** 2)
        pyomo.kernel.pprint(e)
        b = block()
        b.e = e
        pyomo.kernel.pprint(e)
        pyomo.kernel.pprint(b)
        m = block()
        m.b = b
        pyomo.kernel.pprint(e)
        pyomo.kernel.pprint(b)
        pyomo.kernel.pprint(m)

    def test_pickle(self):
        e = self._ctype_factory(expr=1.0)
        self.assertEqual(type(e.expr), float)
        self.assertEqual(e.expr, 1.0)
        self.assertEqual(e.parent, None)
        eup = pickle.loads(pickle.dumps(e))
        self.assertEqual(type(eup.expr), float)
        self.assertEqual(eup.expr, 1.0)
        self.assertEqual(eup.parent, None)
        b = block()
        b.e = e
        self.assertIs(e.parent, b)
        bup = pickle.loads(pickle.dumps(b))
        eup = bup.e
        self.assertEqual(type(eup.expr), float)
        self.assertEqual(eup.expr, 1.0)
        self.assertIs(eup.parent, bup)

    def test_init_no_args(self):
        e = self._ctype_factory()
        self.assertTrue(e.parent is None)
        self.assertEqual(e.ctype, IExpression)
        self.assertTrue(e.expr is None)

    def test_init_args(self):
        e = self._ctype_factory(1.0)
        self.assertTrue(e.parent is None)
        self.assertEqual(e.ctype, IExpression)
        self.assertTrue(e.expr is not None)

    def test_type(self):
        e = self._ctype_factory()
        self.assertTrue(isinstance(e, ICategorizedObject))
        self.assertTrue(isinstance(e, IExpression))
        self.assertTrue(isinstance(e, NumericValue))
        self.assertTrue(isinstance(e, IIdentityExpression))

    def test_call(self):
        e = self._ctype_factory()
        self.assertEqual(e(), None)
        e.expr = 1
        self.assertEqual(e(), 1)
        p = parameter()
        p.value = 2
        e.expr = p + 1
        self.assertEqual(e(), 3)

    def test_is_constant(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(is_constant(e), False)
        e.expr = 1
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(is_constant(e), False)
        p = parameter()
        self.assertEqual(p.is_constant(), False)
        self.assertEqual(is_constant(p), False)
        p.value = 2
        e.expr = p + 1
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(is_constant(e), False)

    def test_is_expression_type(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_expression_type(), True)

    def test_is_parameter_type(self):
        e = self._ctype_factory()
        self.assertEqual(e.is_parameter_type(), False)

    def test_args(self):
        e = self._ctype_factory()
        p = parameter()
        e.expr = p + 1
        self.assertEqual(e.nargs(), 1)
        self.assertTrue(e.arg(0) is e.expr)

    def test_arguments(self):
        e = self._ctype_factory()
        p = parameter()
        e.expr = p + 1
        self.assertEqual(len(tuple(e.args)), 1)
        self.assertTrue(tuple(e.args)[0] is e.expr)

    def test_clone(self):
        e = self._ctype_factory()
        self.assertTrue(e.clone() is e)
        p = parameter()
        e.expr = p
        self.assertTrue(e.clone() is e)
        self.assertTrue(e.clone().expr is p)
        sube = p ** 2 + 1
        e.expr = sube
        self.assertTrue(e.clone() is e)
        self.assertTrue(e.clone().expr is sube)

    def test_division_behavior(self):
        e = self._ctype_factory()
        e.expr = 2
        self.assertIs(type(e.expr), int)
        self.assertEqual((1 / e)(), 0.5)
        self.assertEqual((parameter(1) / e)(), 0.5)
        self.assertEqual(1 / e.expr, 0.5)

    def test_to_string(self):
        b = block()
        e = self._ctype_factory()
        label = str(e)
        self.assertNotEqual(label, None)
        self.assertEqual(e.name, None)
        self.assertEqual(str(e.expr), 'None')
        self.assertEqual(str(e), label)
        self.assertEqual(e.to_string(), label + '{Undefined}')
        self.assertEqual(e.to_string(verbose=False), label + '{Undefined}')
        self.assertEqual(e.to_string(verbose=True), label + '{Undefined}')
        b.e = e
        self.assertNotEqual(e.name, None)
        self.assertEqual(e.to_string(verbose=True), 'e{Undefined}')
        del b.e
        self.assertEqual(e.name, None)
        e.expr = 1
        self.assertEqual(str(e.expr), '1')
        self.assertEqual(str(e), label)
        self.assertEqual(e.to_string(), '1')
        self.assertEqual(e.to_string(verbose=False), '1')
        self.assertEqual(e.to_string(verbose=True), label + '{1}')
        b.e = e
        self.assertNotEqual(e.name, None)
        self.assertEqual(e.to_string(verbose=True), 'e{1}')
        del b.e
        self.assertEqual(e.name, None)
        p = parameter()
        e.expr = p ** 2
        self.assertEqual(str(e.expr), '<parameter>**2')
        self.assertEqual(str(e), label)
        self.assertEqual(e.to_string(), '(<parameter>**2)')
        self.assertEqual(e.to_string(verbose=False), '(<parameter>**2)')
        self.assertEqual(e.to_string(verbose=True), label + '{pow(<parameter>, 2)}')
        b.e = e
        b.p = p
        self.assertNotEqual(e.name, None)
        self.assertNotEqual(p.name, None)
        self.assertEqual(e.to_string(verbose=True), e.name + '{pow(' + p.name + ', 2)}')
        self.assertEqual(e.to_string(verbose=True), 'e{pow(p, 2)}')
        del b.e
        del b.p

    def test_iadd(self):
        e = self._ctype_factory(1.0)
        expr = 0.0
        for v in [1.0, e]:
            expr += v
        self.assertEqual(e.expr, 1)
        self.assertEqual(expr(), 2)
        expr = 0.0
        for v in [e, 1.0]:
            expr += v
        self.assertEqual(e.expr, 1)
        self.assertEqual(expr(), 2)

    def test_isub(self):
        e = self._ctype_factory(1.0)
        expr = 0.0
        for v in [1.0, e]:
            expr -= v
        self.assertEqual(e.expr, 1)
        self.assertEqual(expr(), -2)
        expr = 0.0
        for v in [e, 1.0]:
            expr -= v
        self.assertEqual(e.expr, 1)
        self.assertEqual(expr(), -2)

    def test_imul(self):
        e = self._ctype_factory(3.0)
        expr = 1.0
        for v in [2.0, e]:
            expr *= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 6)
        expr = 1.0
        for v in [e, 2.0]:
            expr *= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 6)

    def test_idiv(self):
        e = self._ctype_factory(3.0)
        expr = e
        for v in [2.0, 1.0]:
            expr /= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 1.5)
        expr = e
        for v in [1.0, 2.0]:
            expr /= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 1.5)
        e = self._ctype_factory(3)
        expr = e
        for v in [2, 1]:
            expr /= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 1.5)
        expr = e
        for v in [1, 2]:
            expr /= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 1.5)

    def test_ipow(self):
        e = self._ctype_factory(3.0)
        expr = e
        for v in [2.0, 1.0]:
            expr **= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 9)
        expr = e
        for v in [1.0, 2.0]:
            expr **= v
        self.assertEqual(e.expr, 3)
        self.assertEqual(expr(), 9)
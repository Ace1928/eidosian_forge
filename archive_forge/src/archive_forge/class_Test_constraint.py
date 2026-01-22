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
class Test_constraint(unittest.TestCase):

    def test_pprint(self):
        v = variable()
        c = constraint((1, v ** 2, 2))
        pprint(c)
        b = block()
        b.c = c
        pprint(c)
        pprint(b)
        m = block()
        m.b = b
        pprint(c)
        pprint(b)
        pprint(m)

    def test_ctype(self):
        c = constraint()
        self.assertIs(c.ctype, IConstraint)
        self.assertIs(type(c), constraint)
        self.assertIs(type(c)._ctype, IConstraint)

    def test_pickle(self):
        c = constraint()
        self.assertIs(c.lb, None)
        self.assertIs(c.body, None)
        self.assertIs(c.ub, None)
        self.assertEqual(c.parent, None)
        cup = pickle.loads(pickle.dumps(c))
        self.assertEqual(cup.lb, None)
        self.assertEqual(cup.body, None)
        self.assertEqual(cup.ub, None)
        self.assertEqual(cup.parent, None)
        b = block()
        b.c = c
        self.assertIs(c.parent, b)
        bup = pickle.loads(pickle.dumps(b))
        cup = bup.c
        self.assertEqual(cup.lb, None)
        self.assertEqual(cup.body, None)
        self.assertEqual(cup.ub, None)
        self.assertIs(cup.parent, bup)

    def test_init(self):
        c = constraint()
        self.assertTrue(c.parent is None)
        self.assertEqual(c.ctype, IConstraint)
        self.assertIs(c.body, None)
        self.assertIs(c.lb, None)
        self.assertIs(c.ub, None)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            self.assertEqual(c(), None)
        with self.assertRaises(ValueError):
            self.assertEqual(c(exception=True), None)
        self.assertEqual(c(exception=False), None)
        self.assertIs(c.slack, None)
        self.assertIs(c.lslack, None)
        self.assertIs(c.uslack, None)

    def test_has_lb_ub(self):
        c = constraint()
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub, None)
        c.lb = float('-inf')
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.has_ub(), False)
        self.assertIs(c.ub, None)
        c.ub = float('inf')
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub, None)
        c.lb = 0
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(type(c.lb), int)
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub, None)
        c.ub = 0
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(type(c.lb), int)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, 0)
        self.assertEqual(type(c.ub), int)
        c.lb = float('inf')
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(type(c.lb), float)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, 0)
        self.assertEqual(type(c.ub), int)
        c.ub = float('-inf')
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(type(c.lb), float)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, float('-inf'))
        self.assertEqual(type(c.ub), float)
        c.rhs = float('inf')
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(type(c.lb), float)
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub, None)
        c.rhs = float('-inf')
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, float('-inf'))
        self.assertEqual(type(c.ub), float)
        c.equality = False
        pL = parameter()
        c.lb = pL
        self.assertIs(c.lower, pL)
        pU = parameter()
        c.ub = pU
        self.assertIs(c.upper, pU)
        with self.assertRaises(ValueError):
            self.assertEqual(c.has_lb(), False)
        self.assertIs(c.lower, pL)
        with self.assertRaises(ValueError):
            self.assertEqual(c.has_ub(), False)
        self.assertIs(c.upper, pU)
        pL.value = float('-inf')
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb, None)
        with self.assertRaises(ValueError):
            self.assertEqual(c.has_ub(), False)
        self.assertIs(c.upper, pU)
        pU.value = float('inf')
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub, None)
        pL.value = 0
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub, None)
        pU.value = 0
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, 0)
        pL.value = float('inf')
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, 0)
        pU.value = float('-inf')
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, float('-inf'))
        pL.value = float('inf')
        c.rhs = pL
        self.assertEqual(c.has_lb(), True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.has_ub(), False)
        self.assertEqual(c.ub, None)
        pL.value = float('-inf')
        c.rhs = pL
        self.assertEqual(c.has_lb(), False)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.has_ub(), True)
        self.assertEqual(c.ub, float('-inf'))

    def test_bounds_getter_setter(self):
        c = constraint()
        self.assertEqual(c.bounds, (None, None))
        self.assertEqual(c.lb, None)
        self.assertEqual(c.ub, None)
        c.bounds = (1, 2)
        self.assertEqual(c.bounds, (1, 2))
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 2)
        c.rhs = 3
        self.assertEqual(c.bounds, (3, 3))
        self.assertEqual(c.lb, 3)
        self.assertEqual(c.ub, 3)
        self.assertEqual(c.rhs, 3)
        with self.assertRaises(ValueError):
            c.bounds = (3, 3)
        self.assertEqual(c.bounds, (3, 3))
        self.assertEqual(c.lb, 3)
        self.assertEqual(c.ub, 3)
        self.assertEqual(c.rhs, 3)
        with self.assertRaises(ValueError):
            c.bounds = (2, 2)
        self.assertEqual(c.bounds, (3, 3))
        self.assertEqual(c.lb, 3)
        self.assertEqual(c.ub, 3)
        self.assertEqual(c.rhs, 3)
        with self.assertRaises(ValueError):
            c.bounds = (1, 2)
        self.assertEqual(c.bounds, (3, 3))
        self.assertEqual(c.lb, 3)
        self.assertEqual(c.ub, 3)
        self.assertEqual(c.rhs, 3)

    def test_init_nonexpr(self):
        v = variable()
        c = constraint(lb=0, body=v, ub=1)
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub, 1)
        with self.assertRaises(ValueError):
            constraint(lb=0, expr=v <= 1)
        with self.assertRaises(ValueError):
            constraint(body=v, expr=v <= 1)
        with self.assertRaises(ValueError):
            constraint(ub=1, expr=v <= 1)
        with self.assertRaises(ValueError):
            constraint(rhs=1, expr=v <= 1)
        c = constraint(expr=v <= 1)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub, 1)
        with self.assertRaises(ValueError):
            constraint(rhs=1, lb=1)
        with self.assertRaises(ValueError):
            constraint(rhs=1, ub=1)
        c = constraint(rhs=1)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertIs(c.body, None)

    def test_type(self):
        c = constraint()
        self.assertTrue(isinstance(c, ICategorizedObject))
        self.assertTrue(isinstance(c, IConstraint))

    def test_active(self):
        c = constraint()
        self.assertEqual(c.active, True)
        c.deactivate()
        self.assertEqual(c.active, False)
        c.activate()
        self.assertEqual(c.active, True)
        b = block()
        self.assertEqual(b.active, True)
        b.deactivate()
        self.assertEqual(b.active, False)
        b.c = c
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, False)
        c.deactivate()
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, False)
        b.activate()
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, True)
        b.activate(shallow=False)
        self.assertEqual(c.active, True)
        self.assertEqual(b.active, True)
        b.deactivate(shallow=False)
        self.assertEqual(c.active, False)
        self.assertEqual(b.active, False)

    def test_equality(self):
        v = variable()
        c = constraint(v == 1)
        self.assertTrue(c.body is not None)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.equality, True)
        c = constraint(1 == v)
        self.assertTrue(c.body is not None)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.equality, True)
        c = constraint(v - 1 == 0)
        self.assertTrue(c.body is not None)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.ub, 0)
        self.assertEqual(c.rhs, 0)
        self.assertEqual(c.equality, True)
        c = constraint(0 == v - 1)
        self.assertTrue(c.body is not None)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.ub, 0)
        self.assertEqual(c.rhs, 0)
        self.assertEqual(c.equality, True)
        c = constraint(rhs=1)
        self.assertIs(c.body, None)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertEqual(c.equality, True)
        with self.assertRaises(ValueError):
            c.lb = 2
        with self.assertRaises(ValueError):
            c.ub = 2
        c.equality = False
        with self.assertRaises(ValueError):
            c.rhs
        self.assertIs(c.body, None)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.equality = True
        c.rhs = 3
        self.assertIs(c.body, None)
        self.assertEqual(c.lb, 3)
        self.assertEqual(c.ub, 3)
        self.assertEqual(c.rhs, 3)
        self.assertEqual(c.equality, True)
        with self.assertRaises(TypeError):
            c.rhs = 'a'
        with self.assertRaises(ValueError):
            c.rhs = None

    def test_nondata_bounds(self):
        c = constraint()
        e = expression()
        eL = expression()
        eU = expression()
        with self.assertRaises(ValueError):
            c.expr = (eL, e, eU)
        e.expr = 1.0
        eL.expr = 1.0
        eU.expr = 1.0
        with self.assertRaises(ValueError):
            c.expr = (eL, e, eU)
        with self.assertRaises(TypeError):
            c.lb = eL
        with self.assertRaises(TypeError):
            c.ub = eU
        vL = variable()
        vU = variable()
        with self.assertRaises(ValueError):
            c.expr = (vL, e, vU)
        with self.assertRaises(TypeError):
            c.lb = vL
        with self.assertRaises(TypeError):
            c.ub = vU
        e.expr = 1.0
        vL.value = 1.0
        vU.value = 1.0
        with self.assertRaises(ValueError):
            c.expr = (vL, e, vU)
        with self.assertRaises(TypeError):
            c.lb = vL
        with self.assertRaises(TypeError):
            c.ub = vU
        with self.assertRaises(TypeError):
            c.rhs = vL
        vL.fixed = True
        vU.fixed = True
        with self.assertRaises(ValueError):
            c.expr = (vL, e, vU)
        with self.assertRaises(TypeError):
            c.lb = vL
        with self.assertRaises(TypeError):
            c.ub = vU
        with self.assertRaises(TypeError):
            c.rhs = vL
        vL.value = 1.0
        vU.value = 1.0
        with self.assertRaises(ValueError):
            c.expr = (vL, 0.0, vU)
        c.body = -2.0
        c.lb = 1.0
        c.ub = 1.0
        self.assertEqual(c.slack, -3.0)
        self.assertEqual(c.lslack, -3.0)
        self.assertEqual(c.uslack, 3.0)
        with self.assertRaises(TypeError):
            c.lb = 'a'
        with self.assertRaises(TypeError):
            c.ub = 'a'
        self.assertEqual(c.lb, 1.0)
        self.assertEqual(c.ub, 1.0)
        vL.value = 2
        vU.value = 1
        c.expr = vL <= vU
        self.assertEqual(c.lb, None)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)
        c.expr = vU >= vL
        self.assertEqual(c.lb, None)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)
        c.expr = vU <= vL
        self.assertEqual(c.lb, None)
        self.assertEqual(c.body(), -1)
        self.assertEqual(c.ub, 0)
        c.expr = vL >= vU
        self.assertEqual(c.lb, None)
        self.assertEqual(c.body(), -1)
        self.assertEqual(c.ub, 0)

    def test_fixed_variable_stays_in_body(self):
        c = constraint()
        x = variable(value=0.5)
        c.expr = (0, x, 1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 0.5)
        self.assertEqual(c.ub, 1)
        x.value = 2
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 2)
        self.assertEqual(c.ub, 1)
        x.fix(0.5)
        c.expr = (0, x, 1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 0.5)
        self.assertEqual(c.ub, 1)
        x.value = 2
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 2)
        self.assertEqual(c.ub, 1)
        x.free()
        x.value = 1
        c.expr = 0 == x
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)
        c.expr = x == 0
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)
        x.fix()
        c.expr = 0 == x
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)
        c.expr = x == 0
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)
        x.free()
        c.expr = 0 == x
        x.fix()
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)
        x.free()
        c.expr = x == 0
        x.fix()
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.ub, 0)

    def test_data_bounds(self):
        c = constraint()
        e = expression(expr=1.0)
        pL = parameter()
        pU = parameter()
        c.expr = (pL, e, pU)
        self.assertIs(c.body, e)
        self.assertIs(c.lower, pL)
        self.assertIs(c.upper, pU)
        e.expr = None
        self.assertIs(c.body, e)
        self.assertIs(c.lower, pL)
        self.assertIs(c.upper, pU)
        c.expr = (pL, e, pU)
        self.assertIs(c.body, e)
        self.assertIs(c.lower, pL)
        self.assertIs(c.upper, pU)
        e.expr = 1.0
        eL = data_expression()
        eU = data_expression()
        c.expr = (eL, e, eU)
        self.assertIs(c.body, e)
        self.assertIs(c.lower, eL)
        self.assertIs(c.upper, eU)
        e.expr = None
        self.assertIs(c.body, e)
        self.assertIs(c.lower, eL)
        self.assertIs(c.upper, eU)
        c.expr = (eL, e, eU)
        self.assertIs(c.body, e)
        self.assertIs(c.lower, eL)
        self.assertIs(c.upper, eU)

    def test_mutable_novalue_param_lower_bound(self):
        x = variable()
        p = parameter()
        p.value = None
        c = constraint(expr=0 <= x - p)
        self.assertEqual(c.equality, False)
        c = constraint(expr=p <= x)
        self.assertIs(c.lower, p)
        self.assertEqual(c.equality, False)
        c = constraint(expr=p <= x + 1)
        self.assertEqual(c.equality, False)
        c = constraint(expr=p + 1 <= x)
        self.assertEqual(c.equality, False)
        c = constraint(expr=(p + 1) ** 2 <= x)
        self.assertEqual(c.equality, False)
        c = constraint(expr=(p, x, p + 1))
        self.assertEqual(c.equality, False)
        c = constraint(expr=x - p >= 0)
        self.assertEqual(c.equality, False)
        c = constraint(expr=x >= p)
        self.assertIs(c.lower, p)
        self.assertEqual(c.equality, False)
        c = constraint(expr=x + 1 >= p)
        self.assertEqual(c.equality, False)
        c = constraint(expr=x >= p + 1)
        self.assertEqual(c.equality, False)
        c = constraint(expr=x >= (p + 1) ** 2)
        self.assertEqual(c.equality, False)
        c = constraint(expr=(p, x, None))
        self.assertIs(c.lower, p)
        self.assertEqual(c.equality, False)
        c = constraint(expr=(p, x + 1, None))
        self.assertEqual(c.equality, False)
        c = constraint(expr=(p + 1, x, None))
        self.assertEqual(c.equality, False)
        c = constraint(expr=(p, x, 1))
        self.assertEqual(c.equality, False)

    def test_mutable_novalue_param_upper_bound(self):
        x = variable()
        p = parameter()
        p.value = None
        c = constraint(expr=x - p <= 0)
        self.assertEqual(c.equality, False)
        c = constraint(expr=x <= p)
        self.assertIs(c.upper, p)
        self.assertEqual(c.equality, False)
        c = constraint(expr=x + 1 <= p)
        self.assertEqual(c.equality, False)
        c = constraint(expr=x <= p + 1)
        self.assertEqual(c.equality, False)
        c = constraint(expr=x <= (p + 1) ** 2)
        self.assertEqual(c.equality, False)
        c = constraint(expr=(p + 1, x, p))
        self.assertEqual(c.equality, False)
        c = constraint(expr=0 >= x - p)
        self.assertEqual(c.equality, False)
        c = constraint(expr=p >= x)
        self.assertIs(c.upper, p)
        self.assertEqual(c.equality, False)
        c = constraint(expr=p >= x + 1)
        self.assertEqual(c.equality, False)
        c = constraint(expr=p + 1 >= x)
        self.assertEqual(c.equality, False)
        c = constraint(expr=(p + 1) ** 2 >= x)
        self.assertEqual(c.equality, False)
        c = constraint(expr=(None, x, p))
        self.assertIs(c.upper, p)
        self.assertEqual(c.equality, False)
        c = constraint(expr=(None, x + 1, p))
        self.assertEqual(c.equality, False)
        c = constraint(expr=(None, x, p + 1))
        self.assertEqual(c.equality, False)
        c = constraint(expr=(1, x, p))
        self.assertEqual(c.equality, False)

    def test_mutable_novalue_param_equality(self):
        x = variable()
        p = parameter()
        p.value = None
        c = constraint(expr=x - p == 0)
        self.assertEqual(c.equality, True)
        c = constraint(expr=x == p)
        self.assertIs(c.upper, p)
        self.assertEqual(c.equality, True)
        c = constraint(expr=x + 1 == p)
        self.assertEqual(c.equality, True)
        c = constraint(expr=x + 1 == (p + 1) ** 2)
        self.assertEqual(c.equality, True)
        c = constraint(expr=x == p + 1)
        self.assertEqual(c.equality, True)
        c = constraint(expr=(x, p))
        self.assertIs(c.upper, p)
        self.assertIs(c.lower, p)
        self.assertIs(c.rhs, p)
        self.assertIs(c.body, x)
        self.assertEqual(c.equality, True)
        c = constraint(expr=(p, x))
        self.assertIs(c.upper, p)
        self.assertIs(c.lower, p)
        self.assertIs(c.rhs, p)
        self.assertIs(c.body, x)
        self.assertEqual(c.equality, True)
        c = constraint(expr=EqualityExpression((p, x)))
        self.assertIs(c.upper, p)
        self.assertIs(c.lower, p)
        self.assertIs(c.rhs, p)
        self.assertIs(c.body, x)
        self.assertEqual(c.equality, True)
        c = constraint(expr=EqualityExpression((x, p)))
        self.assertIs(c.upper, p)
        self.assertIs(c.lower, p)
        self.assertIs(c.rhs, p)
        self.assertIs(c.body, x)
        self.assertEqual(c.equality, True)

    def test_tuple_construct_equality(self):
        x = variable()
        c = constraint((0.0, x))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(type(c.lb), float)
        self.assertIs(c.body, x)
        self.assertEqual(c.ub, 0)
        self.assertEqual(type(c.ub), float)
        c = constraint((x, 0))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertEqual(type(c.lb), int)
        self.assertIs(c.body, x)
        self.assertEqual(c.ub, 0)
        self.assertEqual(type(c.ub), int)

    def test_tuple_construct_inf_equality(self):
        x = variable()
        c = constraint((x, float('inf')))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, None)
        self.assertEqual(c.rhs, float('inf'))
        self.assertEqual(type(c.rhs), float)
        self.assertIs(c.body, x)
        c = constraint((float('inf'), x))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, None)
        self.assertEqual(c.rhs, float('inf'))
        self.assertEqual(type(c.rhs), float)
        self.assertIs(c.body, x)

    def test_tuple_construct_1sided_inequality(self):
        y = variable()
        c = constraint((None, y, 1))
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, 1)
        c = constraint((0, y, None))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, y)
        self.assertIs(c.ub, None)

    def test_tuple_construct_1sided_inf_inequality(self):
        y = variable()
        c = constraint((float('-inf'), y, 1))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, 1)
        self.assertEqual(type(c.ub), int)
        c = constraint((0, y, float('inf')))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, 0)
        self.assertEqual(type(c.lb), int)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, None)

    def test_tuple_construct_unbounded_inequality(self):
        y = variable()
        c = constraint((None, y, None))
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, y)
        self.assertIs(c.ub, None)
        c = constraint((float('-inf'), y, float('inf')))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, None)

    def test_tuple_construct_invalid_1sided_inequality(self):
        x = variable()
        y = variable()
        z = variable()
        with self.assertRaises(ValueError):
            constraint((x, y, None))
        with self.assertRaises(ValueError):
            constraint((None, y, z))

    def test_tuple_construct_2sided_inequality(self):
        y = variable()
        c = constraint((0, y, 1))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, 1)

    def test_construct_invalid_2sided_inequality(self):
        x = variable()
        y = variable()
        z = variable()
        with self.assertRaises(ValueError):
            constraint((x, y, 1))
        with self.assertRaises(ValueError):
            constraint((0, y, z))

    def test_tuple_construct_invalid_2sided_inequality(self):
        x = variable()
        y = variable()
        z = variable()
        with self.assertRaises(ValueError):
            constraint(RangedExpression((x, y, 1), (False, False)))
        with self.assertRaises(ValueError):
            constraint(RangedExpression((0, y, z), (False, False)))

    def test_expr_construct_equality(self):
        x = variable(value=1)
        y = variable(value=1)
        c = constraint(0.0 == x)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, x)
        self.assertEqual(c.ub, 0)
        c = constraint(x == 0.0)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, x)
        self.assertEqual(c.ub, 0)
        c = constraint(x == y)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 0)
        self.assertTrue(c.body is not None)
        self.assertEqual(c(), 0)
        self.assertEqual(c.body(), 0)
        self.assertEqual(c.ub, 0)
        c = constraint()
        c.expr = x == float('inf')
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, None)
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, x)
        c.expr = float('inf') == x
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, None)
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, x)

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

    def test_expr_construct_inf_equality(self):
        x = variable()
        c = constraint(x == float('inf'))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, None)
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, x)
        c = constraint(float('inf') == x)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, None)
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, x)

    def test_expr_construct_1sided_inequality(self):
        y = variable()
        c = constraint(y <= 1)
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, 1)
        c = constraint(0 <= y)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, y)
        self.assertIs(c.ub, None)
        c = constraint(y >= 1)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, 1)
        self.assertIs(c.body, y)
        self.assertIs(c.ub, None)
        c = constraint(0 >= y)
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, 0)

    def test_expr_construct_unbounded_inequality(self):
        y = variable()
        c = constraint(y <= float('inf'))
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, None)
        c = constraint(float('-inf') <= y)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, None)
        self.assertIs(c.body, y)
        self.assertIs(c.ub, None)
        c = constraint(y >= float('-inf'))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, None)
        self.assertIs(c.body, y)
        self.assertIs(c.ub, None)
        c = constraint(float('inf') >= y)
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, y)
        self.assertEqual(c.ub, None)

    def test_expr_construct_unbounded_inequality(self):
        y = variable()
        c = constraint(y <= float('-inf'))
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertEqual(c.ub, float('-inf'))
        self.assertIs(c.body, y)
        c = constraint(float('inf') <= y)
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, float('inf'))
        self.assertIs(c.ub, None)
        self.assertIs(c.body, y)
        c = constraint(y >= float('inf'))
        self.assertEqual(c.equality, False)
        self.assertEqual(c.lb, float('inf'))
        self.assertIs(c.ub, None)
        self.assertIs(c.body, y)
        c = constraint(float('-inf') >= y)
        self.assertEqual(c.equality, False)
        self.assertIs(c.lb, None)
        self.assertEqual(c.ub, float('-inf'))
        self.assertIs(c.body, y)

    def test_expr_invalid_double_sided_inequality(self):
        x = variable()
        y = variable()
        c = constraint()
        c.expr = (0, x - y, 1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.expr = (x, x - y, 1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.expr = (0, x - y, y)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.expr = (1, x - y, x)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)
        with self.assertRaises(ValueError):
            c.expr = (y, x - y, 0)

    def test_equality_infinite(self):
        c = constraint()
        v = variable()
        c.expr = v == 1
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, 1)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.rhs, 1)
        self.assertIs(c.body, v)
        c.expr = v == float('inf')
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, None)
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, v)
        c.expr = (v, float('inf'))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, None)
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, v)
        c.expr = float('inf') == v
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, None)
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, v)
        c.expr = (float('inf'), v)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, float('inf'))
        self.assertEqual(c.ub, None)
        self.assertEqual(c.rhs, float('inf'))
        self.assertIs(c.body, v)
        c.expr = v == float('-inf')
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.ub, float('-inf'))
        self.assertEqual(c.rhs, float('-inf'))
        self.assertIs(c.body, v)
        c.expr = (v, float('-inf'))
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.ub, float('-inf'))
        self.assertEqual(c.rhs, float('-inf'))
        self.assertIs(c.body, v)
        c.expr = float('-inf') == v
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.ub, float('-inf'))
        self.assertEqual(c.rhs, float('-inf'))
        self.assertIs(c.body, v)
        c.expr = (float('-inf'), v)
        self.assertEqual(c.equality, True)
        self.assertEqual(c.lb, None)
        self.assertEqual(c.ub, float('-inf'))
        self.assertEqual(c.rhs, float('-inf'))
        self.assertIs(c.body, v)

    def test_equality_nonnumeric(self):
        c = constraint()
        v = variable()
        c.expr = v == 1
        with self.assertRaises(TypeError):
            c.expr = (v, 'x')
        with self.assertRaises(TypeError):
            c.expr = ('x', v)

    def test_slack_methods(self):
        x = variable(value=2)
        L = 1
        U = 5
        cE = constraint(rhs=L, body=x)
        x.value = 4
        self.assertEqual(cE.body(), 4)
        self.assertEqual(cE.slack, -3)
        self.assertEqual(cE.lslack, 3)
        self.assertEqual(cE.uslack, -3)
        x.value = 6
        self.assertEqual(cE.body(), 6)
        self.assertEqual(cE.slack, -5)
        self.assertEqual(cE.lslack, 5)
        self.assertEqual(cE.uslack, -5)
        x.value = 0
        self.assertEqual(cE.body(), 0)
        self.assertEqual(cE.slack, -1)
        self.assertEqual(cE.lslack, -1)
        self.assertEqual(cE.uslack, 1)
        x.value = None
        with self.assertRaises(ValueError):
            cE.body()
        self.assertEqual(cE.body(exception=False), None)
        self.assertEqual(cE.slack, None)
        self.assertEqual(cE.lslack, None)
        self.assertEqual(cE.uslack, None)
        cE = constraint(rhs=U, body=x)
        x.value = 4
        self.assertEqual(cE.body(), 4)
        self.assertEqual(cE.slack, -1)
        self.assertEqual(cE.lslack, -1)
        self.assertEqual(cE.uslack, 1)
        x.value = 6
        self.assertEqual(cE.body(), 6)
        self.assertEqual(cE.slack, -1)
        self.assertEqual(cE.lslack, 1)
        self.assertEqual(cE.uslack, -1)
        x.value = 0
        self.assertEqual(cE.body(), 0)
        self.assertEqual(cE.slack, -5)
        self.assertEqual(cE.lslack, -5)
        self.assertEqual(cE.uslack, 5)
        x.value = None
        with self.assertRaises(ValueError):
            cE.body()
        self.assertEqual(cE.body(exception=False), None)
        self.assertEqual(cE.slack, None)
        self.assertEqual(cE.lslack, None)
        self.assertEqual(cE.uslack, None)
        cL = constraint(lb=L, body=x)
        x.value = 4
        self.assertEqual(cL.body(), 4)
        self.assertEqual(cL.slack, 3)
        self.assertEqual(cL.lslack, 3)
        self.assertEqual(cL.uslack, float('inf'))
        x.value = 6
        self.assertEqual(cL.body(), 6)
        self.assertEqual(cL.slack, 5)
        self.assertEqual(cL.lslack, 5)
        self.assertEqual(cL.uslack, float('inf'))
        x.value = 0
        self.assertEqual(cL.body(), 0)
        self.assertEqual(cL.slack, -1)
        self.assertEqual(cL.lslack, -1)
        self.assertEqual(cL.uslack, float('inf'))
        x.value = None
        with self.assertRaises(ValueError):
            cL.body()
        self.assertEqual(cL.body(exception=False), None)
        self.assertEqual(cL.slack, None)
        self.assertEqual(cL.lslack, None)
        self.assertEqual(cL.uslack, None)
        cL = constraint(lb=float('-inf'), body=x)
        x.value = 4
        self.assertEqual(cL.body(), 4)
        self.assertEqual(cL.slack, float('inf'))
        self.assertEqual(cL.lslack, float('inf'))
        self.assertEqual(cL.uslack, float('inf'))
        x.value = 6
        self.assertEqual(cL.body(), 6)
        self.assertEqual(cL.slack, float('inf'))
        self.assertEqual(cL.lslack, float('inf'))
        self.assertEqual(cL.uslack, float('inf'))
        x.value = 0
        self.assertEqual(cL.body(), 0)
        self.assertEqual(cL.slack, float('inf'))
        self.assertEqual(cL.lslack, float('inf'))
        self.assertEqual(cL.uslack, float('inf'))
        x.value = None
        with self.assertRaises(ValueError):
            cL.body()
        self.assertEqual(cL.body(exception=False), None)
        self.assertEqual(cL.slack, None)
        self.assertEqual(cL.lslack, None)
        self.assertEqual(cL.uslack, None)
        cU = constraint(body=x, ub=U)
        x.value = 4
        self.assertEqual(cU.body(), 4)
        self.assertEqual(cU.slack, 1)
        self.assertEqual(cU.lslack, float('inf'))
        self.assertEqual(cU.uslack, 1)
        x.value = 6
        self.assertEqual(cU.body(), 6)
        self.assertEqual(cU.slack, -1)
        self.assertEqual(cU.lslack, float('inf'))
        self.assertEqual(cU.uslack, -1)
        x.value = 0
        self.assertEqual(cU.body(), 0)
        self.assertEqual(cU.slack, 5)
        self.assertEqual(cU.lslack, float('inf'))
        self.assertEqual(cU.uslack, 5)
        x.value = None
        with self.assertRaises(ValueError):
            cU.body()
        self.assertEqual(cU.body(exception=False), None)
        self.assertEqual(cU.slack, None)
        self.assertEqual(cU.lslack, None)
        self.assertEqual(cU.uslack, None)
        cU = constraint(body=x, ub=float('inf'))
        x.value = 4
        self.assertEqual(cU.body(), 4)
        self.assertEqual(cU.slack, float('inf'))
        self.assertEqual(cU.lslack, float('inf'))
        self.assertEqual(cU.uslack, float('inf'))
        x.value = 6
        self.assertEqual(cU.body(), 6)
        self.assertEqual(cU.slack, float('inf'))
        self.assertEqual(cU.lslack, float('inf'))
        self.assertEqual(cU.uslack, float('inf'))
        x.value = 0
        self.assertEqual(cU.body(), 0)
        self.assertEqual(cU.slack, float('inf'))
        self.assertEqual(cU.lslack, float('inf'))
        self.assertEqual(cU.uslack, float('inf'))
        x.value = None
        with self.assertRaises(ValueError):
            cU.body()
        self.assertEqual(cU.body(exception=False), None)
        self.assertEqual(cU.slack, None)
        self.assertEqual(cU.lslack, None)
        self.assertEqual(cU.uslack, None)
        cR = constraint(lb=L, body=x, ub=U)
        x.value = 4
        self.assertEqual(cR.body(), 4)
        self.assertEqual(cR.slack, 1)
        self.assertEqual(cR.lslack, 3)
        self.assertEqual(cR.uslack, 1)
        x.value = 6
        self.assertEqual(cR.body(), 6)
        self.assertEqual(cR.slack, -1)
        self.assertEqual(cR.lslack, 5)
        self.assertEqual(cR.uslack, -1)
        x.value = 0
        self.assertEqual(cR.body(), 0)
        self.assertEqual(cR.slack, -1)
        self.assertEqual(cR.lslack, -1)
        self.assertEqual(cR.uslack, 5)
        x.value = None
        with self.assertRaises(ValueError):
            cR.body()
        self.assertEqual(cR.body(exception=False), None)
        self.assertEqual(cR.slack, None)
        self.assertEqual(cR.lslack, None)
        self.assertEqual(cR.uslack, None)
        cR = constraint(body=x)
        x.value = 4
        self.assertEqual(cR.body(), 4)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = 6
        self.assertEqual(cR.body(), 6)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = 0
        self.assertEqual(cR.body(), 0)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = None
        with self.assertRaises(ValueError):
            cR.body()
        self.assertEqual(cR.body(exception=False), None)
        self.assertEqual(cR.slack, None)
        self.assertEqual(cR.lslack, None)
        self.assertEqual(cR.uslack, None)
        cR = constraint(body=x, lb=float('-inf'), ub=float('inf'))
        x.value = 4
        self.assertEqual(cR.body(), 4)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = 6
        self.assertEqual(cR.body(), 6)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = 0
        self.assertEqual(cR.body(), 0)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = None
        with self.assertRaises(ValueError):
            cR.body()
        self.assertEqual(cR.body(exception=False), None)
        self.assertEqual(cR.slack, None)
        self.assertEqual(cR.lslack, None)
        self.assertEqual(cR.uslack, None)
        cR = constraint(body=x, lb=parameter(L), ub=parameter(U))
        x.value = 4
        self.assertEqual(cR.body(), 4)
        self.assertEqual(cR.slack, 1)
        self.assertEqual(cR.lslack, 3)
        self.assertEqual(cR.uslack, 1)
        x.value = 6
        self.assertEqual(cR.body(), 6)
        self.assertEqual(cR.slack, -1)
        self.assertEqual(cR.lslack, 5)
        self.assertEqual(cR.uslack, -1)
        x.value = 0
        self.assertEqual(cR.body(), 0)
        self.assertEqual(cR.slack, -1)
        self.assertEqual(cR.lslack, -1)
        self.assertEqual(cR.uslack, 5)
        x.value = None
        with self.assertRaises(ValueError):
            cR.body()
        self.assertEqual(cR.body(exception=False), None)
        self.assertEqual(cR.slack, None)
        self.assertEqual(cR.lslack, None)
        self.assertEqual(cR.uslack, None)
        cR = constraint(body=x, lb=parameter(float('-inf')), ub=parameter(float('inf')))
        x.value = 4
        self.assertEqual(cR.body(), 4)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = 6
        self.assertEqual(cR.body(), 6)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = 0
        self.assertEqual(cR.body(), 0)
        self.assertEqual(cR.slack, float('inf'))
        self.assertEqual(cR.lslack, float('inf'))
        self.assertEqual(cR.uslack, float('inf'))
        x.value = None
        with self.assertRaises(ValueError):
            cR.body()
        self.assertEqual(cR.body(exception=False), None)
        self.assertEqual(cR.slack, None)
        self.assertEqual(cR.lslack, None)
        self.assertEqual(cR.uslack, None)

    def test_expr(self):
        x = variable(value=1.0)
        c = constraint()
        c.expr = (0, x, 2)
        self.assertEqual(c(), 1)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.lb, 0)
        self.assertEqual(c.ub, 2)
        self.assertEqual(c.equality, False)
        c.expr = (-2, x, 0)
        self.assertEqual(c(), 1)
        self.assertEqual(c.body(), 1)
        self.assertEqual(c.lb, -2)
        self.assertEqual(c.ub, 0)
        self.assertEqual(c.equality, False)

    def test_expr_getter(self):
        c = constraint()
        self.assertIs(c.expr, None)
        v = variable()
        c.expr = 0 <= v
        self.assertIsNot(c.expr, None)
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, v)
        self.assertIs(c.ub, None)
        self.assertEqual(c.equality, False)
        c.expr = v <= 1
        self.assertIsNot(c.expr, None)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)
        c.expr = (0, v, 1)
        self.assertIsNot(c.expr, None)
        self.assertEqual(c.lb, 0)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, False)
        c.expr = v == 1
        self.assertIsNot(c.expr, None)
        self.assertEqual(c.lb, 1)
        self.assertIs(c.body, v)
        self.assertEqual(c.ub, 1)
        self.assertEqual(c.equality, True)
        c.expr = None
        self.assertIs(c.expr, None)
        self.assertIs(c.lb, None)
        self.assertIs(c.body, None)
        self.assertIs(c.ub, None)
        self.assertEqual(c.equality, False)

    def test_expr_wrong_type(self):
        c = constraint()
        with self.assertRaises(ValueError):
            c.expr = 2
        with self.assertRaises(ValueError):
            c.expr = True

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

    def test_expression_constructor_coverage(self):
        x = variable()
        y = variable()
        z = variable()
        L = parameter(value=0)
        U = parameter(value=1)
        expr = U >= x
        expr = expr >= L
        c = constraint(expr)
        expr = x <= z
        expr = expr >= y
        with self.assertRaises(ValueError):
            constraint(expr)
        expr = x >= z
        expr = y >= expr
        with self.assertRaises(ValueError):
            constraint(expr)
        expr = y <= x
        expr = y >= expr
        with self.assertRaises(ValueError):
            constraint(expr)
        L.value = 0
        c = constraint(x >= L)
        U.value = 0
        c = constraint(U >= x)
        L.value = 0
        U.value = 1
        expr = U <= x
        expr = expr <= L
        c = constraint(expr)
        expr = x >= z
        expr = expr <= y
        with self.assertRaises(ValueError):
            constraint(expr)
        expr = x <= z
        expr = y <= expr
        with self.assertRaises(ValueError):
            constraint(expr)
        expr = y >= x
        expr = y <= expr
        with self.assertRaises(ValueError):
            constraint(expr)
        L.value = 0
        expr = x <= L
        c = constraint(expr)
        U.value = 0
        expr = U <= x
        c = constraint(expr)
        x = variable()
        with self.assertRaises(ValueError):
            constraint(x + x)
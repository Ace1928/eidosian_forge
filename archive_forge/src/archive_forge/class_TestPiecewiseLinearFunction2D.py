from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
class TestPiecewiseLinearFunction2D(unittest.TestCase):

    def make_ln_x_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 10))
        m.f = f
        m.f1 = f1
        m.f2 = f2
        m.f3 = f3
        return m

    def check_ln_x_approx(self, pw, x):
        self.assertEqual(len(pw._simplices), 3)
        self.assertEqual(len(pw._linear_functions), 3)
        simplices = [(0, 1), (1, 2), (2, 3)]
        for idx, simplex in enumerate(simplices):
            self.assertEqual(pw._simplices[idx], simplices[idx])
        assertExpressionsEqual(self, pw._linear_functions[0](x), log(3) / 2 * x - log(3) / 2, places=7)
        assertExpressionsEqual(self, pw._linear_functions[1](x), log(2) / 3 * x + log(3 / 2), places=7)
        assertExpressionsEqual(self, pw._linear_functions[2](x), log(5 / 3) / 4 * x + log(6 / (5 / 3) ** (3 / 2)), places=7)

    def check_x_squared_approx(self, pw, x):
        self.assertEqual(len(pw._simplices), 3)
        self.assertEqual(len(pw._linear_functions), 3)
        simplices = [(0, 1), (1, 2), (2, 3)]
        for idx, simplex in enumerate(simplices):
            self.assertEqual(pw._simplices[idx], simplices[idx])
        assertExpressionsStructurallyEqual(self, pw._linear_functions[0](x), 4 * x - 3, places=7)
        assertExpressionsStructurallyEqual(self, pw._linear_functions[1](x), 9 * x - 18, places=7)
        assertExpressionsStructurallyEqual(self, pw._linear_functions[2](x), 16 * x - 60, places=7)

    def test_pw_linear_approx_of_ln_x_simplices(self):
        m = self.make_ln_x_model()
        simplices = [(1, 3), (3, 6), (6, 10)]
        m.pw = PiecewiseLinearFunction(simplices=simplices, function=m.f)
        self.check_ln_x_approx(m.pw, m.x)

    def test_pw_linear_approx_of_ln_x_points(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(points=[1, 3, 6, 10], function=m.f)
        self.check_ln_x_approx(m.pw, m.x)

    def test_pw_linear_approx_of_ln_x_linear_funcs(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(simplices=[(1, 3), (3, 6), (6, 10)], linear_functions=[m.f1, m.f2, m.f3])
        self.check_ln_x_approx(m.pw, m.x)

    def test_pw_linear_approx_of_ln_x_tabular_data(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(tabular_data={1: 0, 3: log(3), 6: log(6), 10: log(10)})
        self.check_ln_x_approx(m.pw, m.x)

    def test_use_pw_function_in_constraint(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(simplices=[(1, 3), (3, 6), (6, 10)], linear_functions=[m.f1, m.f2, m.f3])
        m.c = Constraint(expr=m.pw(m.x) <= 1)
        self.assertEqual(str(m.c.body.expr), 'pw(x)')

    def test_evaluate_pw_function(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(simplices=[(1, 3), (3, 6), (6, 10)], linear_functions=[m.f1, m.f2, m.f3])
        self.assertAlmostEqual(m.pw(1), 0)
        self.assertAlmostEqual(m.pw(2), m.f1(2))
        self.assertAlmostEqual(m.pw(3), log(3))
        self.assertAlmostEqual(m.pw(4.5), m.f2(4.5))
        self.assertAlmostEqual(m.pw(9.2), m.f3(9.2))
        self.assertAlmostEqual(m.pw(10), log(10))

    def test_indexed_pw_linear_function_approximate_over_simplices(self):
        m = self.make_ln_x_model()
        m.z = Var([1, 2], bounds=(-10, 10))

        def g1(x):
            return x ** 2

        def g2(x):
            return log(x)
        m.funcs = {1: g1, 2: g2}
        simplices = [(1, 3), (3, 6), (6, 10)]
        m.pw = PiecewiseLinearFunction([1, 2], simplices=simplices, function_rule=lambda m, i: m.funcs[i])
        self.check_ln_x_approx(m.pw[2], m.z[2])
        self.check_x_squared_approx(m.pw[1], m.z[1])

    def test_indexed_pw_linear_function_approximate_over_points(self):
        m = self.make_ln_x_model()
        m.z = Var([1, 2], bounds=(-10, 10))

        def g1(x):
            return x ** 2

        def g2(x):
            return log(x)
        m.funcs = {1: g1, 2: g2}

        def silly_pts_rule(m, i):
            return [1, 3, 6, 10]
        m.pw = PiecewiseLinearFunction([1, 2], points=silly_pts_rule, function_rule=lambda m, i: m.funcs[i])
        self.check_ln_x_approx(m.pw[2], m.z[2])
        self.check_x_squared_approx(m.pw[1], m.z[1])

    def test_indexed_pw_linear_function_tabular_data(self):
        m = self.make_ln_x_model()
        m.z = Var([1, 2], bounds=(-10, 10))

        def silly_tabular_data_rule(m, i):
            if i == 1:
                return {1: 1, 3: 9, 6: 36, 10: 100}
            if i == 2:
                return {1: 0, 3: log(3), 6: log(6), 10: log(10)}
        m.pw = PiecewiseLinearFunction([1, 2], tabular_data_rule=silly_tabular_data_rule)
        self.check_ln_x_approx(m.pw[2], m.z[2])
        self.check_x_squared_approx(m.pw[1], m.z[1])

    def test_indexed_pw_linear_function_linear_funcs_and_simplices(self):
        m = self.make_ln_x_model()
        m.z = Var([1, 2], bounds=(-10, 10))

        def silly_simplex_rule(m, i):
            return [(1, 3), (3, 6), (6, 10)]

        def h1(x):
            return 4 * x - 3

        def h2(x):
            return 9 * x - 18

        def h3(x):
            return 16 * x - 60

        def silly_linear_func_rule(m, i):
            return [h1, h2, h3]
        m.pw = PiecewiseLinearFunction([1, 2], simplices=silly_simplex_rule, linear_functions=silly_linear_func_rule)
        self.check_x_squared_approx(m.pw[1], m.z[1])
        self.check_x_squared_approx(m.pw[2], m.z[2])

    def test_pickle(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(points=[1, 3, 6, 10], function=m.f)
        m.c = Constraint(expr=m.pw(m.x) >= 0.35)
        unpickle = pickle.loads(pickle.dumps(m))
        m_buf = StringIO()
        m.pprint(ostream=m_buf)
        m_output = m_buf.getvalue()
        unpickle_buf = StringIO()
        unpickle.pprint(ostream=unpickle_buf)
        unpickle_output = unpickle_buf.getvalue()
        self.assertMultiLineEqual(m_output, unpickle_output)
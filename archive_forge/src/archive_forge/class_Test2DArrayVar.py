import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import IntegerSet
from pyomo.core.expr.numeric_expr import (
from pyomo.core.staleflag import StaleFlagManager
from pyomo.environ import (
from pyomo.core.base.units_container import units, pint_available, UnitsError
class Test2DArrayVar(TestSimpleVar):

    def setUp(self):
        PyomoModel.setUp(self)
        self.model.A = Set(initialize=[1, 2])

    def test_fixed_attr(self):
        """Test fixed attribute"""
        self.model.x = Var(self.model.A, self.model.A)
        self.model.y = Var(self.model.A, self.model.A)
        self.instance = self.model.create_instance()
        self.instance.x.fixed = True
        self.assertEqual(self.instance.x[1, 2].fixed, False)
        self.instance.y[1, 2].fixed = True
        self.assertEqual(self.instance.y[1, 2].fixed, True)

    def test_value_attr(self):
        """Test value attribute"""
        self.model.x = Var(self.model.A, self.model.A, dense=True)
        self.model.y = Var(self.model.A, self.model.A, dense=True)
        self.instance = self.model.create_instance()
        try:
            self.instance.x = 3.5
            self.fail('Expected ValueError')
        except ValueError:
            pass
        self.instance.y[1, 2] = 3.5
        self.assertEqual(self.instance.y[1, 2].value, 3.5)

    def test_initialize_with_function(self):
        """Test initialize option with an initialization rule"""

        def init_rule(model, key1, key2):
            i = key1 + 1
            return key1 == 1 and 1.3 or 2.3
        self.model.x = Var(self.model.A, self.model.A, initialize=init_rule)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1, 1].value, 1.3)
        self.assertEqual(self.instance.x[2, 2].value, 2.3)
        self.instance.x[1, 1] = 1
        self.instance.x[2, 2] = 2
        self.assertEqual(self.instance.x[1, 1].value, 1)
        self.assertEqual(self.instance.x[2, 2].value, 2)

    def test_initialize_with_dict(self):
        """Test initialize option with a dictionary"""
        self.model.x = Var(self.model.A, self.model.A, initialize={(1, 1): 1.3, (2, 2): 2.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1, 1].value, 1.3)
        self.assertEqual(self.instance.x[2, 2].value, 2.3)
        self.instance.x[1, 1] = 1
        self.instance.x[2, 2] = 2
        self.assertEqual(self.instance.x[1, 1].value, 1)
        self.assertEqual(self.instance.x[2, 2].value, 2)

    def test_initialize_with_const(self):
        """Test initialize option with a constant"""
        self.model.x = Var(self.model.A, self.model.A, initialize=3)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1, 1].value, 3)
        self.assertEqual(self.instance.x[2, 2].value, 3)
        self.instance.x[1, 1] = 1
        self.instance.x[2, 2] = 2
        self.assertEqual(self.instance.x[1, 1].value, 1)
        self.assertEqual(self.instance.x[2, 2].value, 2)

    def test_without_initial_value(self):
        """Test default initialization"""
        self.model.x = Var(self.model.A, self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1, 1].value, None)
        self.assertEqual(self.instance.x[2, 2].value, None)
        self.instance.x[1, 1] = 5
        self.instance.x[2, 2] = 6
        self.assertEqual(self.instance.x[1, 1].value, 5)
        self.assertEqual(self.instance.x[2, 2].value, 6)

    def test_initialize_option(self):
        """Test initialize option"""
        self.model.x = Var(self.model.A, self.model.A, initialize={(1, 1): 1.3, (2, 2): 2.3})
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1, 1].value, 1.3)
        self.assertEqual(self.instance.x[2, 2].value, 2.3)
        try:
            value(self.instance.x[1, 2])
            self.fail('Expected ValueError')
        except ValueError:
            pass

    def test_bounds_option1(self):
        """Test bounds option"""

        def x_bounds(model, i, j):
            return (-1.0 * (i + j), 1.0 * (i + j))
        self.model.x = Var(self.model.A, self.model.A, bounds=x_bounds)
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x[1, 1].lb), -2.0)
        self.assertEqual(value(self.instance.x[1, 2].ub), 3.0)

    def test_bounds_option2(self):
        """Test bounds option"""
        self.model.x = Var(self.model.A, self.model.A, bounds=(-1.0, 1.0))
        self.instance = self.model.create_instance()
        self.assertEqual(value(self.instance.x[1, 1].lb), -1.0)
        self.assertEqual(value(self.instance.x[1, 1].ub), 1.0)

    def test_rule_option(self):
        """Test rule option"""

        def x_init(model, i, j):
            return 1.3
        self.model.x = Var(self.model.A, self.model.A, initialize=x_init)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x[1, 2].value, 1.3)

    def test_dim(self):
        """Test dim method"""
        self.model.x = Var(self.model.A, self.model.A)
        self.instance = self.model.create_instance()
        self.assertEqual(self.instance.x.dim(), 2)

    def test_keys(self):
        """Test keys method"""
        self.model.x = Var(self.model.A, self.model.A, dense=True)
        self.instance = self.model.create_instance()
        ans = [(1, 1), (1, 2), (2, 1), (2, 2)]
        self.assertEqual(list(sorted(self.instance.x.keys())), ans)

    def test_len(self):
        """Test len method"""
        self.model.x = Var(self.model.A, self.model.A, dense=True)
        self.instance = self.model.create_instance()
        self.assertEqual(len(self.instance.x), 4)

    def test_value(self):
        """Check the value of the variable"""
        self.model.x = Var(self.model.A, self.model.A, initialize=3.3)
        self.instance = self.model.create_instance()
        tmp = value(self.instance.x[1, 1].value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = float(self.instance.x[1, 1].value)
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 3.3)
        tmp = int(self.instance.x[1, 1].value)
        self.assertEqual(type(tmp), int)
        self.assertEqual(tmp, 3)
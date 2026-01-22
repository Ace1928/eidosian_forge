import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
class TestArrayCon(unittest.TestCase):

    def create_model(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2, 3, 4])
        return model

    def test_rule_option1(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(model.A, rule=f)
        self.assertEqual(model.c[1](), 8)
        self.assertEqual(model.c[2](), 16)
        self.assertEqual(len(model.c), 4)

    def test_rule_option2(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            if i % 2 == 0:
                return Constraint.Skip
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(model.A, rule=f)
        self.assertEqual(model.c[1](), 8)
        self.assertEqual(len(model.c), 2)

    def test_rule_option3(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            if i % 2 == 0:
                return Constraint.Skip
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(model.A, rule=f)
        self.assertEqual(model.c[1](), 8)
        self.assertEqual(len(model.c), 2)

    def test_rule_option2a(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        @simple_constraint_rule
        def f(model, i):
            if i % 2 == 0:
                return None
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(model.A, rule=f)
        self.assertEqual(model.c[1](), 8)
        self.assertEqual(len(model.c), 2)

    def test_rule_option3a(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        @simple_constraint_rule
        def f(model, i):
            if i % 2 == 0:
                return None
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(model.A, rule=f)
        self.assertEqual(model.c[1](), 8)
        self.assertEqual(len(model.c), 2)

    def test_dim(self):
        model = self.create_model()
        model.c = Constraint(model.A)
        self.assertEqual(model.c.dim(), 1)

    def test_keys(self):
        model = self.create_model()
        model.c = Constraint(model.A)
        self.assertEqual(len(list(model.c.keys())), 0)

    def test_len(self):
        model = self.create_model()
        model.c = Constraint(model.A)
        self.assertEqual(len(model.c), 0)
        model = self.create_model()
        model.B = RangeSet(1, 4)
        'Test rule option'

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            ans = ans == 2
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)
        self.assertEqual(len(model.c), 1)

    def test_setitem(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(range(5))
        self.assertEqual(len(m.c), 0)
        m.c[2] = m.x ** 2 <= 4
        self.assertEqual(len(m.c), 1)
        self.assertEqual(list(m.c.keys()), [2])
        self.assertIsInstance(m.c[2], _GeneralConstraintData)
        self.assertEqual(m.c[2].upper, 4)
        m.c[3] = Constraint.Skip
        self.assertEqual(len(m.c), 1)
        self.assertRaisesRegex(KeyError, '3', m.c.__getitem__, 3)
        self.assertRaisesRegex(ValueError, "'c\\[3\\]': rule returned None", m.c.__setitem__, 3, None)
        self.assertEqual(len(m.c), 1)
        m.c[2] = Constraint.Skip
        self.assertEqual(len(m.c), 0)
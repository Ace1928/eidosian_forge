import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
class Test2DArrayObj(unittest.TestCase):

    def create_model(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2])
        return model

    def test_rule_option1(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i, k):
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            return ans
        model.x = Var(model.B, initialize=2)
        model.obj = Objective(model.A, model.A, rule=f)
        try:
            self.assertEqual(model.obj(), None)
            self.fail('Expected TypeError')
        except TypeError:
            pass
        self.assertEqual(model.obj[1, 1](), 8)
        self.assertEqual(model.obj[2, 1](), 16)
        self.assertEqual(value(model.obj[1, 1]), 8)
        self.assertEqual(value(model.obj[2, 1]), 16)

    def test_sense_option(self):
        """Test sense option"""
        model = self.create_model()
        model.obj1 = Objective(model.A, model.A, rule=lambda m, i, j: 1.0, sense=maximize)
        model.obj2 = Objective(model.A, model.A, rule=lambda m, i, j: 1.0, sense=minimize)
        model.obj3 = Objective(model.A, model.A, rule=lambda m, i, j: 1.0)
        self.assertTrue(len(model.A) > 0)
        self.assertEqual(len(model.obj1), len(model.A) * len(model.A))
        self.assertEqual(len(model.obj2), len(model.A) * len(model.A))
        self.assertEqual(len(model.obj3), len(model.A) * len(model.A))
        for i in model.A:
            for j in model.A:
                self.assertEqual(model.obj1[i, j].sense, maximize)
                self.assertEqual(model.obj1[i, j].is_minimizing(), False)
                self.assertEqual(model.obj2[i, j].sense, minimize)
                self.assertEqual(model.obj2[i, j].is_minimizing(), True)
                self.assertEqual(model.obj3[i, j].sense, minimize)
                self.assertEqual(model.obj3[i, j].is_minimizing(), True)

    def test_dim(self):
        """Test dim method"""
        model = self.create_model()
        model.obj = Objective(model.A, model.A)
        self.assertEqual(model.obj.dim(), 2)

    def test_keys(self):
        """Test keys method"""
        model = self.create_model()

        def A_rule(model, i, j):
            return model.x
        model.x = Var()
        model.obj = Objective(model.A, model.A, rule=A_rule)
        self.assertEqual(len(list(model.obj.keys())), 4)

    def test_len(self):
        """Test len method"""
        model = self.create_model()
        model.obj = Objective(model.A, model.A)
        self.assertEqual(len(model.obj), 0)
        model = self.create_model()
        'Test rule option'

        def f(model):
            ans = 0
            for i in model.x.keys():
                ans = ans + model.x[i]
            return ans
        model.x = Var(RangeSet(1, 4), initialize=2)
        model.obj = Objective(rule=f)
        self.assertEqual(len(model.obj), 1)
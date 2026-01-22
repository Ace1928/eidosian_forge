import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
class TestObjList(unittest.TestCase):

    def create_model(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2, 3, 4])
        return model

    def test_conlist_skip(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = ObjectiveList()
        self.assertTrue(1 not in model.c)
        self.assertEqual(len(model.c), 0)
        model.c.add(Objective.Skip)
        self.assertTrue(1 not in model.c)
        self.assertEqual(len(model.c), 0)
        model.c.add(model.x + 1)
        self.assertTrue(1 not in model.c)
        self.assertTrue(2 in model.c)
        self.assertEqual(len(model.c), 1)

    def test_rule_option1(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            if i > 4:
                return ObjectiveList.End
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            return ans
        model.x = Var(model.B, initialize=2)
        model.o = ObjectiveList(rule=f)
        self.assertEqual(model.o[1](), 8)
        self.assertEqual(model.o[2](), 16)
        self.assertEqual(len(model.o), 4)

    def test_rule_option2(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            if i > 2:
                return ObjectiveList.End
            i = 2 * i - 1
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            return ans
        model.x = Var(model.B, initialize=2)
        model.o = ObjectiveList(rule=f)
        self.assertEqual(model.o[1](), 8)
        self.assertEqual(len(model.o), 2)

    def test_rule_option1a(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        @simple_objectivelist_rule
        def f(model, i):
            if i > 4:
                return None
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            return ans
        model.x = Var(model.B, initialize=2)
        model.o = ObjectiveList(rule=f)
        self.assertEqual(model.o[1](), 8)
        self.assertEqual(model.o[2](), 16)
        self.assertEqual(len(model.o), 4)

    def test_rule_option2a(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        @simple_objectivelist_rule
        def f(model, i):
            if i > 2:
                return None
            i = 2 * i - 1
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            return ans
        model.x = Var(model.B, initialize=2)
        model.o = ObjectiveList(rule=f)
        self.assertEqual(model.o[1](), 8)
        self.assertEqual(len(model.o), 2)

    def test_rule_option3(self):
        """Test rule option"""
        model = self.create_model()
        model.y = Var(initialize=2)

        def f(model):
            yield model.y
            yield (2 * model.y)
            yield (2 * model.y)
            yield ObjectiveList.End
        model.c = ObjectiveList(rule=f)
        self.assertEqual(len(model.c), 3)
        self.assertEqual(model.c[1](), 2)
        model.d = ObjectiveList(rule=f(model))
        self.assertEqual(len(model.d), 3)
        self.assertEqual(model.d[1](), 2)

    def test_rule_option4(self):
        """Test rule option"""
        model = self.create_model()
        model.y = Var(initialize=2)
        model.c = ObjectiveList(rule=((i + 1) * model.y for i in range(3)))
        self.assertEqual(len(model.c), 3)
        self.assertEqual(model.c[1](), 2)

    def test_dim(self):
        """Test dim method"""
        model = self.create_model()
        model.o = ObjectiveList()
        self.assertEqual(model.o.dim(), 1)

    def test_keys(self):
        """Test keys method"""
        model = self.create_model()
        model.o = ObjectiveList()
        self.assertEqual(len(list(model.o.keys())), 0)

    def test_len(self):
        """Test len method"""
        model = self.create_model()
        model.o = ObjectiveList()
        self.assertEqual(len(model.o), 0)
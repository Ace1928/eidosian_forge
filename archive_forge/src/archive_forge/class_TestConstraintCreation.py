import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
class TestConstraintCreation(unittest.TestCase):

    def create_model(self, abstract=False):
        if abstract is True:
            model = AbstractModel()
        else:
            model = ConcreteModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        return model

    def test_tuple_construct_equality(self):
        model = self.create_model()

        def rule(model):
            return (0.0, model.x)
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, True)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.x)
        self.assertEqual(model.c.upper, 0)
        model = self.create_model()

        def rule(model):
            return (model.x, 0.0)
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, True)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.x)
        self.assertEqual(model.c.upper, 0)

    def test_tuple_construct_inf_equality(self):
        model = self.create_model(abstract=True)

        def rule(model):
            return (model.x, float('inf'))
        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)
        model = self.create_model(abstract=True)

        def rule(model):
            return (float('inf'), model.x)
        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)

    def test_tuple_construct_1sided_inequality(self):
        model = self.create_model()

        def rule(model):
            return (None, model.y, 1)
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, 1)
        model = self.create_model()

        def rule(model):
            return (0, model.y, None)
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

    def test_tuple_construct_1sided_inf_inequality(self):
        model = self.create_model()

        def rule(model):
            return (float('-inf'), model.y, 1)
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, 1)
        model = self.create_model()

        def rule(model):
            return (0, model.y, float('inf'))
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

    def test_tuple_construct_unbounded_inequality(self):
        model = self.create_model()

        def rule(model):
            return (None, model.y, None)
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)
        model = self.create_model()

        def rule(model):
            return (float('-inf'), model.y, float('inf'))
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

    def test_tuple_construct_invalid_1sided_inequality(self):
        model = self.create_model(abstract=True)

        def rule(model):
            return (model.x, model.y, None)
        model.c = Constraint(rule=rule)
        instance = model.create_instance()
        self.assertEqual(instance.c.lower, None)
        self.assertIsInstance(instance.c.body, SumExpression)
        self.assertEqual(instance.c.upper, 0)
        model = self.create_model(abstract=True)

        def rule(model):
            return (None, model.y, model.z)
        model.c = Constraint(rule=rule)
        instance = model.create_instance()
        self.assertEqual(instance.c.lower, None)
        self.assertIsInstance(instance.c.body, SumExpression)
        self.assertEqual(instance.c.upper, 0)

    def test_tuple_construct_2sided_inequality(self):
        model = self.create_model()

        def rule(model):
            return (0, model.y, 1)
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, 1)

    def test_tuple_construct_invalid_2sided_inequality(self):
        model = self.create_model(abstract=True)

        def rule(model):
            return (model.x, model.y, 1)
        model.c = Constraint(rule=rule)
        instance = model.create_instance()
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable lower bound"):
            instance.c.lower
        self.assertIs(instance.c.body, instance.y)
        self.assertEqual(instance.c.upper, 1)
        instance.x.fix(3)
        self.assertEqual(value(instance.c.lower), 3)
        model = self.create_model(abstract=True)

        def rule(model):
            return (0, model.y, model.z)
        model.c = Constraint(rule=rule)
        instance = model.create_instance()
        self.assertEqual(instance.c.lower, 0)
        self.assertIs(instance.c.body, instance.y)
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable upper bound"):
            instance.c.upper
        instance.z.fix(3)
        self.assertEqual(value(instance.c.upper), 3)

    def test_expr_construct_equality(self):
        model = self.create_model()

        def rule(model):
            return 0.0 == model.x
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, True)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.x)
        self.assertEqual(model.c.upper, 0)
        model = self.create_model()

        def rule(model):
            return model.x == 0.0
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, True)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.x)
        self.assertEqual(model.c.upper, 0)

    def test_expr_construct_inf_equality(self):
        model = self.create_model(abstract=True)

        def rule(model):
            return model.x == float('inf')
        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)
        model = self.create_model(abstract=True)

        def rule(model):
            return float('inf') == model.x
        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)

    def test_expr_construct_1sided_inequality(self):
        model = self.create_model()

        def rule(model):
            return model.y <= 1
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, 1)
        model = self.create_model()

        def rule(model):
            return 0 <= model.y
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)
        model = self.create_model()

        def rule(model):
            return model.y >= 1
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, 1)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)
        model = self.create_model()

        def rule(model):
            return 0 >= model.y
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, 0)

    def test_expr_construct_unbounded_inequality(self):
        model = self.create_model()

        def rule(model):
            return model.y <= float('inf')
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)
        model = self.create_model()

        def rule(model):
            return float('-inf') <= model.y
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)
        model = self.create_model()

        def rule(model):
            return model.y >= float('-inf')
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)
        model = self.create_model()

        def rule(model):
            return float('inf') >= model.y
        model.c = Constraint(rule=rule)
        self.assertEqual(model.c.equality, False)
        self.assertEqual(model.c.lower, None)
        self.assertIs(model.c.body, model.y)
        self.assertEqual(model.c.upper, None)

    def test_expr_construct_invalid_unbounded_inequality(self):
        model = self.create_model(abstract=True)

        def rule(model):
            return model.y <= float('-inf')
        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)
        model = self.create_model(abstract=True)

        def rule(model):
            return float('inf') <= model.y
        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)
        model = self.create_model(abstract=True)

        def rule(model):
            return model.y >= float('inf')
        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)
        model = self.create_model(abstract=True)

        def rule(model):
            return float('-inf') >= model.y
        model.c = Constraint(rule=rule)
        self.assertRaises(ValueError, model.create_instance)

    def test_expr_construct_invalid(self):
        m = ConcreteModel()
        c = Constraint(rule=lambda m: None)
        self.assertRaisesRegex(ValueError, '.*rule returned None', m.add_component, 'c', c)
        m = ConcreteModel()
        c = Constraint([1], rule=lambda m, i: None)
        self.assertRaisesRegex(ValueError, '.*rule returned None', m.add_component, 'c', c)
        m = ConcreteModel()
        c = Constraint(rule=lambda m: True)
        self.assertRaisesRegex(ValueError, '.*resolved to a trivial Boolean \\(True\\).*Constraint\\.Feasible', m.add_component, 'c', c)
        m = ConcreteModel()
        c = Constraint([1], rule=lambda m, i: True)
        self.assertRaisesRegex(ValueError, '.*resolved to a trivial Boolean \\(True\\).*Constraint\\.Feasible', m.add_component, 'c', c)
        m = ConcreteModel()
        c = Constraint(rule=lambda m: False)
        self.assertRaisesRegex(ValueError, '.*resolved to a trivial Boolean \\(False\\).*Constraint\\.Infeasible', m.add_component, 'c', c)
        m = ConcreteModel()
        c = Constraint([1], rule=lambda m, i: False)
        self.assertRaisesRegex(ValueError, '.*resolved to a trivial Boolean \\(False\\).*Constraint\\.Infeasible', m.add_component, 'c', c)

    def test_nondata_bounds(self):
        model = ConcreteModel()
        model.c = Constraint()
        model.v = Var([1, 2, 3])
        model.e1 = Expression()
        model.e2 = Expression()
        model.e3 = Expression()
        model.c.set_value((model.e1, model.e2, model.e3))
        self.assertIsNone(model.c._lower)
        self.assertIsNone(model.c._body)
        self.assertIsNone(model.c._upper)
        self.assertIs(model.c.lower, model.e1)
        self.assertIs(model.c.body, model.e2)
        self.assertIs(model.c.upper, model.e3)
        model.e1.expr = 1
        model.e2.expr = 2
        model.e3.expr = 3
        self.assertEqual(value(model.c.lower), 1)
        self.assertEqual(value(model.c.body), 2)
        self.assertEqual(value(model.c.upper), 3)
        model.e1 = model.v[1]
        model.e2 = model.v[2]
        model.e3 = model.v[3]
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable lower bound"):
            model.c.lower
        self.assertIs(model.c.body.expr, model.v[2])
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable upper bound"):
            model.c.upper

    def test_mutable_novalue_param_lower_bound(self):
        model = ConcreteModel()
        model.x = Var()
        model.p = Param(mutable=True)
        model.p.value = None
        model.c = Constraint(expr=0 <= model.x - model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.p <= model.x)
        self.assertTrue(model.c.lower is model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.p <= model.x + 1)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.p + 1 <= model.x)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=(model.p + 1) ** 2 <= model.x)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=(model.p, model.x, model.p + 1))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x - model.p >= 0)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x >= model.p)
        self.assertTrue(model.c.lower is model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x + 1 >= model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x >= model.p + 1)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x >= (model.p + 1) ** 2)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=(model.p, model.x, None))
        self.assertTrue(model.c.lower is model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=(model.p, model.x + 1, None))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=(model.p + 1, model.x, None))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=(model.p, model.x, 1))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

    def test_mutable_novalue_param_upper_bound(self):
        model = ConcreteModel()
        model.x = Var()
        model.p = Param(mutable=True)
        model.p.value = None
        model.c = Constraint(expr=model.x - model.p <= 0)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x <= model.p)
        self.assertTrue(model.c.upper is model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x + 1 <= model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x <= model.p + 1)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x <= (model.p + 1) ** 2)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=(model.p + 1, model.x, model.p))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=0 >= model.x - model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.p >= model.x)
        self.assertTrue(model.c.upper is model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.p >= model.x + 1)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=model.p + 1 >= model.x)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=(model.p + 1) ** 2 >= model.x)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=(None, model.x, model.p))
        self.assertTrue(model.c.upper is model.p)
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=(None, model.x + 1, model.p))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=(None, model.x, model.p + 1))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)
        model.c = Constraint(expr=(1, model.x, model.p))
        self.assertEqual(model.c.equality, False)
        model.del_component(model.c)

    def test_mutable_novalue_param_equality(self):
        model = ConcreteModel()
        model.x = Var()
        model.p = Param(mutable=True)
        model.p.value = None
        model.c = Constraint(expr=model.x - model.p == 0)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x == model.p)
        self.assertTrue(model.c.upper is model.p)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x + 1 == model.p)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x + 1 == (model.p + 1) ** 2)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)
        model.c = Constraint(expr=model.x == model.p + 1)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)
        model.c = Constraint(expr=inequality(model.p, model.x, model.p))
        self.assertTrue(model.c.upper is model.p)
        model.del_component(model.c)
        model.c = Constraint(expr=(model.x, model.p))
        self.assertTrue(model.c.upper is model.p)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)
        model.c = Constraint(expr=(model.p, model.x))
        self.assertTrue(model.c.upper is model.p)
        self.assertEqual(model.c.equality, True)
        model.del_component(model.c)

    def test_inequality(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=inequality(lower=-1, body=m.x))
        self.assertEqual(m.c.lower.value, -1)
        self.assertIs(m.c.body, m.x)
        self.assertIs(m.c.upper, None)
        del m.c
        m.c = Constraint(expr=inequality(body=m.x, upper=1))
        self.assertIs(m.c.lower, None)
        self.assertIs(m.c.body, m.x)
        self.assertEqual(m.c.upper.value, 1)
        del m.c
        m.c = Constraint(expr=inequality(lower=-1, body=m.x, upper=1))
        self.assertEqual(m.c.lower.value, -1)
        self.assertIs(m.c.body, m.x)
        self.assertEqual(m.c.upper.value, 1)
import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
class TestTemplatizeRule(unittest.TestCase):

    def test_simple_rule(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.x = Var(m.I)

        @m.Constraint(m.I)
        def c(m, i):
            return m.x[i] <= 0
        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 1)
        self.assertIs(indices[0]._set, m.I)
        self.assertEqual(str(template), 'x[_1]  <=  0')
        self.assertEqual(list(m.I), list(range(1, 4)))
        indices[0].set_value(2)
        self.assertEqual(str(resolve_template(template)), 'x[2]  <=  0')

    def test_tuple_rules(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.x = Var(m.I)

        @m.Constraint(m.I)
        def c(m, i):
            return (None, m.x[i], 0)
        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 1)
        self.assertIs(indices[0]._set, m.I)
        self.assertEqual(str(template), 'x[_1]  <=  0')
        self.assertEqual(list(m.I), list(range(1, 4)))
        indices[0].set_value(2)
        self.assertEqual(str(resolve_template(template)), 'x[2]  <=  0')

        @m.Constraint(m.I)
        def d(m, i):
            return (0, m.x[i], 10)
        template, indices = templatize_constraint(m.d)
        self.assertEqual(len(indices), 1)
        self.assertIs(indices[0]._set, m.I)
        self.assertEqual(str(template), '0  <=  x[_1]  <=  10')
        self.assertEqual(list(m.I), list(range(1, 4)))
        indices[0].set_value(2)
        self.assertEqual(str(resolve_template(template)), '0  <=  x[2]  <=  10')

        @m.Constraint(m.I)
        def e(m, i):
            return (m.x[i], 0)
        template, indices = templatize_constraint(m.e)
        self.assertEqual(len(indices), 1)
        self.assertIs(indices[0]._set, m.I)
        self.assertEqual(str(template), 'x[_1]  ==  0')
        self.assertEqual(list(m.I), list(range(1, 4)))
        indices[0].set_value(2)
        self.assertEqual(str(resolve_template(template)), 'x[2]  ==  0')

    def test_simple_rule_nonfinite_set(self):
        m = ConcreteModel()
        m.x = Var(Integers, dense=False)

        @m.Constraint(Integers)
        def c(m, i):
            return m.x[i] <= 0
        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 1)
        self.assertIs(indices[0]._set, Integers)
        self.assertEqual(str(template), 'x[_1]  <=  0')
        indices[0].set_value(2)
        self.assertEqual(str(resolve_template(template)), 'x[2]  <=  0')

    def test_simple_abstract_rule(self):
        m = AbstractModel()
        m.I = RangeSet(3)
        m.x = Var(m.I)

        @m.Constraint(m.I)
        def c(m, i):
            return m.x[i] <= 0
        with self.assertRaisesRegex(ValueError, '.*has not been constructed'):
            template, indices = templatize_constraint(m.c)
        m.I.construct()
        m.x.construct()
        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 1)
        self.assertIs(indices[0]._set, m.I)
        self.assertEqual(str(template), 'x[_1]  <=  0')

    def test_simple_sum_rule(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.J = RangeSet(3)
        m.x = Var(m.I, m.J)

        @m.Constraint(m.I)
        def c(m, i):
            return sum((m.x[i, j] for j in m.J)) <= 0
        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 1)
        self.assertIs(indices[0]._set, m.I)
        self.assertEqual(template.to_string(verbose=True), 'templatesum(getitem(x, _1, _2), iter(_2, J))  <=  0')
        self.assertEqual(str(template), 'SUM(x[_1,_2] for _2 in J)  <=  0')
        indices[0].set_value(2)
        self.assertEqual(str(resolve_template(template)), 'x[2,1] + x[2,2] + x[2,3]  <=  0')

    def test_nested_sum_rule(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.J = RangeSet(3)
        m.K = Set(m.I, initialize={1: [10], 2: [10, 20], 3: [10, 20, 30]})
        m.x = Var(m.I, m.J, [10, 20, 30])

        @m.Constraint()
        def c(m):
            return sum((sum((m.x[i, j, k] for k in m.K[i])) for j in m.J for i in m.I)) <= 0
        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 0)
        self.assertEqual(template.to_string(verbose=True), 'templatesum(templatesum(getitem(x, _2, _1, _3), iter(_3, getitem(K, _2))), iter(_1, J), iter(_2, I))  <=  0')
        self.assertEqual(str(template), 'SUM(SUM(x[_2,_1,_3] for _3 in K[_2]) for _1 in J for _2 in I)  <=  0')
        self.assertEqual(str(resolve_template(template)), 'x[1,1,10] + (x[2,1,10] + x[2,1,20]) + (x[3,1,10] + x[3,1,20] + x[3,1,30]) + (x[1,2,10]) + (x[2,2,10] + x[2,2,20]) + (x[3,2,10] + x[3,2,20] + x[3,2,30]) + (x[1,3,10]) + (x[2,3,10] + x[2,3,20]) + (x[3,3,10] + x[3,3,20] + x[3,3,30])  <=  0')

    def test_multidim_nested_sum_rule(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.J = RangeSet(3)
        m.JI = m.J * m.I
        m.K = Set(m.I, initialize={1: [10], 2: [10, 20], 3: [10, 20, 30]})
        m.x = Var(m.I, m.J, [10, 20, 30])

        @m.Constraint()
        def c(m):
            return sum((sum((m.x[i, j, k] for k in m.K[i])) for j, i in m.JI)) <= 0
        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 0)
        self.assertEqual(template.to_string(verbose=True), 'templatesum(templatesum(getitem(x, _2, _1, _3), iter(_3, getitem(K, _2))), iter(_1, _2, JI))  <=  0')
        self.assertEqual(str(template), 'SUM(SUM(x[_2,_1,_3] for _3 in K[_2]) for _1, _2 in JI)  <=  0')
        self.assertEqual(str(resolve_template(template)), 'x[1,1,10] + (x[2,1,10] + x[2,1,20]) + (x[3,1,10] + x[3,1,20] + x[3,1,30]) + (x[1,2,10]) + (x[2,2,10] + x[2,2,20]) + (x[3,2,10] + x[3,2,20] + x[3,2,30]) + (x[1,3,10]) + (x[2,3,10] + x[2,3,20]) + (x[3,3,10] + x[3,3,20] + x[3,3,30])  <=  0')

    def test_multidim_nested_sum_rule(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.J = RangeSet(3)
        m.JI = m.J * m.I
        m.K = Set(m.I, initialize={1: [10], 2: [10, 20], 3: [10, 20, 30]})
        m.x = Var(m.I, m.J, [10, 20, 30])

        @m.Constraint()
        def c(m):
            return sum((sum((m.x[i, j, k] for k in m.K[i])) for j, i in m.JI)) <= 0
        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 0)
        self.assertEqual(template.to_string(verbose=True), 'templatesum(templatesum(getitem(x, _2, _1, _3), iter(_3, getitem(K, _2))), iter(_1, _2, JI))  <=  0')
        self.assertEqual(str(template), 'SUM(SUM(x[_2,_1,_3] for _3 in K[_2]) for _1, _2 in JI)  <=  0')
        self.assertEqual(str(resolve_template(template)), 'x[1,1,10] + (x[2,1,10] + x[2,1,20]) + (x[3,1,10] + x[3,1,20] + x[3,1,30]) + (x[1,2,10]) + (x[2,2,10] + x[2,2,20]) + (x[3,2,10] + x[3,2,20] + x[3,2,30]) + (x[1,3,10]) + (x[2,3,10] + x[2,3,20]) + (x[3,3,10] + x[3,3,20] + x[3,3,30])  <=  0')

    def test_multidim_nested_getattr_sum_rule(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.J = RangeSet(3)
        m.JI = m.J * m.I
        m.K = Set(m.I, initialize={1: [10], 2: [10, 20], 3: [10, 20, 30]})
        m.x = Var(m.I, m.J, [10, 20, 30])

        @m.Block(m.I)
        def b(b, i):
            b.K = RangeSet(10, 10 * i, 10)

        @m.Constraint()
        def c(m):
            return sum((sum((m.x[i, j, k] for k in m.b[i].K)) for j, i in m.JI)) <= 0
        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 0)
        self.assertEqual(template.to_string(verbose=True), "templatesum(templatesum(getitem(x, _2, _1, _3), iter(_3, getattr(getitem(b, _2), 'K'))), iter(_1, _2, JI))  <=  0")
        self.assertEqual(str(template), 'SUM(SUM(x[_2,_1,_3] for _3 in b[_2].K) for _1, _2 in JI)  <=  0')
        self.assertEqual(str(resolve_template(template)), 'x[1,1,10] + (x[2,1,10] + x[2,1,20]) + (x[3,1,10] + x[3,1,20] + x[3,1,30]) + (x[1,2,10]) + (x[2,2,10] + x[2,2,20]) + (x[3,2,10] + x[3,2,20] + x[3,2,30]) + (x[1,3,10]) + (x[2,3,10] + x[2,3,20]) + (x[3,3,10] + x[3,3,20] + x[3,3,30])  <=  0')

    def test_eval_getattr(self):
        m = ConcreteModel()
        m.T = RangeSet(3)

        @m.Block(m.T)
        def b(b, i):
            b.x = Var(initialize=i)

            @b.Block(m.T)
            def bb(bb, j):
                bb.I = RangeSet(i * j)
                bb.y = Var(bb.I, initialize=lambda m, i: i)
        t = IndexTemplate(m.T)
        e = m.b[t].x
        with self.assertRaisesRegex(ValueError, 'Evaluating uninitialized IndexTemplate \\({T}\\)'):
            value(e())
        with self.assertRaisesRegex(KeyError, "Index 'None' is not valid for indexed component 'b'"):
            self.assertIsNone(e(exception=False))
        with self.assertRaisesRegex(KeyError, "Index 'None' is not valid for indexed component 'b'"):
            self.assertIsNone(e(False))
        t.set_value(2)
        self.assertEqual(e(), 2)
        f = e.set_value(5)
        self.assertIs(f.__class__, CallExpression)
        self.assertEqual(f._kwds, ())
        self.assertEqual(len(f._args_), 2)
        self.assertIs(f._args_[0].__class__, EXPR.Structural_GetAttrExpression)
        self.assertIs(f._args_[0]._args_[0], e)
        self.assertEqual(f._args_[1], 5)
        self.assertEqual(value(m.b[2].x), 2)
        f()
        self.assertEqual(value(m.b[2].x), 5)
        f = e.set_value('a', skip_validation=True)
        self.assertIs(f.__class__, CallExpression)
        self.assertEqual(f._kwds, ('skip_validation',))
        self.assertEqual(len(f._args_), 3)
        self.assertIs(f._args_[0].__class__, EXPR.Structural_GetAttrExpression)
        self.assertIs(f._args_[0]._args_[0], e)
        self.assertEqual(f._args_[1], 'a')
        self.assertEqual(f._args_[2], True)
        f()
        self.assertEqual(value(m.b[2].x), 'a')
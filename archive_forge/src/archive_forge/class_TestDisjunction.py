from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core import ConcreteModel, Var, Constraint
from pyomo.gdp import Disjunction, Disjunct
from pyomo.gdp.disjunct import AutoLinkedBooleanVar, AutoLinkedBinaryVar
class TestDisjunction(unittest.TestCase):

    def test_empty_disjunction(self):
        m = ConcreteModel()
        m.d = Disjunct()
        m.e = Disjunct()
        m.x1 = Disjunction()
        self.assertEqual(len(m.x1), 0)
        m.x1 = [m.d, m.e]
        self.assertEqual(len(m.x1), 1)
        self.assertEqual(m.x1.disjuncts, [m.d, m.e])
        m.x2 = Disjunction([1, 2, 3, 4])
        self.assertEqual(len(m.x2), 0)
        m.x2[2] = [m.d, m.e]
        self.assertEqual(len(m.x2), 1)
        self.assertEqual(m.x2[2].disjuncts, [m.d, m.e])

    def test_construct_implicit_disjuncts(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.d = Disjunction(expr=[m.x <= 0, m.y >= 1])
        self.assertEqual(len(m.component_map(Disjunction)), 1)
        self.assertEqual(len(m.component_map(Disjunct)), 1)
        implicit_disjuncts = list(m.component_map(Disjunct).keys())
        self.assertEqual(implicit_disjuncts[0][:2], 'd_')
        disjuncts = m.d.disjuncts
        self.assertEqual(len(disjuncts), 2)
        self.assertIs(disjuncts[0].parent_block(), m)
        self.assertIs(disjuncts[0].constraint[1].body, m.x)
        self.assertIs(disjuncts[1].parent_block(), m)
        self.assertIs(disjuncts[1].constraint[1].body, m.y)
        m.add_component('e_disjuncts', Var())
        m.e = Disjunction(expr=[m.y <= 0, m.x >= 1])
        self.assertEqual(len(m.component_map(Disjunction)), 2)
        self.assertEqual(len(m.component_map(Disjunct)), 2)
        implicit_disjuncts = list(m.component_map(Disjunct).keys())
        self.assertEqual(implicit_disjuncts[1][:12], 'e_disjuncts_')
        disjuncts = m.e.disjuncts
        self.assertEqual(len(disjuncts), 2)
        self.assertIs(disjuncts[0].parent_block(), m)
        self.assertIs(disjuncts[0].constraint[1].body, m.y)
        self.assertIs(disjuncts[1].parent_block(), m)
        self.assertIs(disjuncts[1].constraint[1].body, m.x)
        self.assertEqual(len(disjuncts[0].parent_component().name), 13)
        self.assertEqual(disjuncts[0].name[:12], 'e_disjuncts_')

        def _gen():
            yield (m.y <= 4)
            yield (m.x >= 5)
        m.f = Disjunction(expr=[[m.y <= 0, m.x >= 1], (m.y <= 2, m.x >= 3), _gen()])
        self.assertEqual(len(m.component_map(Disjunction)), 3)
        self.assertEqual(len(m.component_map(Disjunct)), 3)
        implicit_disjuncts = list(m.component_map(Disjunct).keys())
        self.assertEqual(implicit_disjuncts[2][:12], 'f_disjuncts')
        disjuncts = m.f.disjuncts
        self.assertEqual(len(disjuncts), 3)
        self.assertIs(disjuncts[0].parent_block(), m)
        self.assertIs(disjuncts[0].constraint[1].body, m.y)
        self.assertEqual(disjuncts[0].constraint[1].upper, 0)
        self.assertIs(disjuncts[0].constraint[2].body, m.x)
        self.assertEqual(disjuncts[0].constraint[2].lower, 1)
        self.assertIs(disjuncts[1].parent_block(), m)
        self.assertIs(disjuncts[1].constraint[1].body, m.y)
        self.assertEqual(disjuncts[1].constraint[1].upper, 2)
        self.assertIs(disjuncts[1].constraint[2].body, m.x)
        self.assertEqual(disjuncts[1].constraint[2].lower, 3)
        self.assertIs(disjuncts[2].parent_block(), m)
        self.assertIs(disjuncts[2].constraint[1].body, m.y)
        self.assertEqual(disjuncts[2].constraint[1].upper, 4)
        self.assertIs(disjuncts[2].constraint[2].body, m.x)
        self.assertEqual(disjuncts[2].constraint[2].lower, 5)
        self.assertEqual(len(disjuncts[0].parent_component().name), 11)
        self.assertEqual(disjuncts[0].name, 'f_disjuncts[0]')

    def test_construct_invalid_component(self):
        m = ConcreteModel()
        m.d = Disjunct([1, 2])
        with self.assertRaisesRegex(ValueError, "Unexpected term for Disjunction 'dd'.\n    Expected a Disjunct object, relational expression, or iterable of\n    relational expressions but got 'IndexedDisjunct'"):
            m.dd = Disjunction(expr=[m.d])
        with self.assertRaisesRegex(ValueError, "Unexpected term for Disjunction 'ee'.\n    Expected a Disjunct object, relational expression, or iterable of\n    relational expressions but got 'str' in 'list'"):
            m.ee = Disjunction(expr=[['a']])
        with self.assertRaisesRegex(ValueError, "Unexpected term for Disjunction 'ff'.\n    Expected a Disjunct object, relational expression, or iterable of\n    relational expressions but got 'str'"):
            m.ff = Disjunction(expr=['a'])
import pickle
import os
import io
import sys
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.relational_expr import (
class TestGenerate_RelationalExpression(unittest.TestCase):

    def setUp(self):
        m = AbstractModel()
        m.I = Set()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.x = Var(m.I)
        self.m = m

    def tearDown(self):
        self.m = None

    def test_simpleEquality(self):
        m = self.m
        e = m.a == m.b
        self.assertIs(type(e), EqualityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)

    def test_equalityErrors(self):
        m = self.m
        e = m.a == m.b
        with self.assertRaisesRegex(TypeError, 'Attempting to use a non-numeric type \\(EqualityExpression\\) in a numeric expression context.'):
            e == m.a
        with self.assertRaisesRegex(TypeError, 'Attempting to use a non-numeric type \\(EqualityExpression\\) in a numeric expression context.'):
            m.a == e
        with self.assertRaisesRegex(TypeError, 'Argument .* is an indexed numeric value'):
            m.x == m.a
        with self.assertRaisesRegex(TypeError, 'Argument .* is an indexed numeric value'):
            m.a == m.x

    def test_simpleInequality1(self):
        m = self.m
        e = m.a < m.b
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e._strict, True)
        e = m.a <= m.b
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e._strict, False)
        e = m.a > m.b
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e._strict, True)
        e = m.a >= m.b
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e._strict, False)

    def test_simpleInequality2(self):
        m = self.m
        e = inequality(lower=m.a, body=m.b, strict=True)
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e._strict, True)
        e = inequality(lower=m.a, upper=m.b)
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e._strict, False)
        e = inequality(lower=m.b, upper=m.a, strict=True)
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e._strict, True)
        e = m.a >= m.b
        e = inequality(body=m.b, upper=m.a)
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e._strict, False)
        try:
            inequality(None, None)
            self.fail('expected invalid inequality error.')
        except ValueError:
            pass
        try:
            inequality(m.a, None)
            self.fail('expected invalid inequality error.')
        except ValueError:
            pass
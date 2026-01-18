import pickle
import os
import io
import sys
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.relational_expr import (
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
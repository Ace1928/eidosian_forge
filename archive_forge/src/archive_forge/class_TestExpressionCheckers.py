import json
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import (
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.fileutils import import_file
import os
from os.path import abspath, dirname, normpath, join
class TestExpressionCheckers(unittest.TestCase):
    """
    Class for testing the pyomo.DAE simulator expression checkers.
    """

    def setUp(self):
        """
        Setting up testing model
        """
        self.m = m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)

    def test_check_getitemexpression(self):
        m = self.m
        t = IndexTemplate(m.t)
        e = m.dv[t] == m.v[t]
        temp = _check_getitemexpression(e, 0)
        self.assertIs(e.arg(0), temp[0])
        self.assertIs(e.arg(1), temp[1])
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(m.v, temp[1].arg(0))
        temp = _check_getitemexpression(e, 1)
        self.assertIsNone(temp)
        e = m.v[t] == m.dv[t]
        temp = _check_getitemexpression(e, 1)
        self.assertIs(e.arg(0), temp[1])
        self.assertIs(e.arg(1), temp[0])
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(m.v, temp[1].arg(0))
        temp = _check_getitemexpression(e, 0)
        self.assertIsNone(temp)
        e = m.v[t] == m.v[t]
        temp = _check_getitemexpression(e, 0)
        self.assertIsNone(temp)
        temp = _check_getitemexpression(e, 1)
        self.assertIsNone(temp)

    def test_check_productexpression(self):
        m = self.m
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        m.z = Var()
        t = IndexTemplate(m.t)
        e = 5 * m.dv[t] == m.v[t]
        temp = _check_productexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        e = m.v[t] == 5 * m.dv[t]
        temp = _check_productexpression(e, 1)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        e = m.p * m.dv[t] == m.v[t]
        temp = _check_productexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        e = m.v[t] == m.p * m.dv[t]
        temp = _check_productexpression(e, 1)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        e = m.mp * m.dv[t] == m.v[t]
        temp = _check_productexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(m.mp, temp[1].arg(1))
        self.assertIs(e.arg(1), temp[1].arg(0))
        e = m.v[t] == m.mp * m.dv[t]
        temp = _check_productexpression(e, 1)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(m.mp, temp[1].arg(1))
        self.assertIs(e.arg(0), temp[1].arg(0))
        e = m.y * m.dv[t] / m.z == m.v[t]
        temp = _check_productexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(e.arg(1), temp[1].arg(0).arg(0))
        self.assertIs(m.z, temp[1].arg(0).arg(1))
        e = m.v[t] == m.y * m.dv[t] / m.z
        temp = _check_productexpression(e, 1)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(e.arg(0), temp[1].arg(0).arg(0))
        self.assertIs(m.z, temp[1].arg(0).arg(1))
        e = m.y / (m.dv[t] * m.z) == m.mp
        temp = _check_productexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(m.y, temp[1].arg(0))
        self.assertIs(e.arg(1), temp[1].arg(1).arg(0))
        e = m.mp == m.y / (m.dv[t] * m.z)
        temp = _check_productexpression(e, 1)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(m.y, temp[1].arg(0))
        self.assertIs(e.arg(0), temp[1].arg(1).arg(0))
        e = m.v[t] * m.y / m.z == m.v[t] * m.y / m.z
        temp = _check_productexpression(e, 0)
        self.assertIsNone(temp)
        temp = _check_productexpression(e, 1)
        self.assertIsNone(temp)

    def test_check_negationexpression(self):
        m = self.m
        t = IndexTemplate(m.t)
        e = -m.dv[t] == m.v[t]
        temp = _check_negationexpression(e, 0)
        self.assertIs(e.arg(0).arg(0), temp[0])
        self.assertIs(e.arg(1), temp[1].arg(0))
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(m.v, temp[1].arg(0).arg(0))
        temp = _check_negationexpression(e, 1)
        self.assertIsNone(temp)
        e = m.v[t] == -m.dv[t]
        temp = _check_negationexpression(e, 1)
        self.assertIs(e.arg(0), temp[1].arg(0))
        self.assertIs(e.arg(1).arg(0), temp[0])
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(m.v, temp[1].arg(0).arg(0))
        temp = _check_negationexpression(e, 0)
        self.assertIsNone(temp)
        e = -m.v[t] == -m.v[t]
        temp = _check_negationexpression(e, 0)
        self.assertIsNone(temp)
        temp = _check_negationexpression(e, 1)
        self.assertIsNone(temp)

    def test_check_viewsumexpression(self):
        m = self.m
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        m.z = Var()
        t = IndexTemplate(m.t)
        e = m.dv[t] + m.y + m.z == m.v[t]
        temp = _check_viewsumexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.SumExpression)
        self.assertIs(type(temp[1].arg(0)), EXPR.Numeric_GetItemExpression)
        self.assertIs(type(temp[1].arg(1)), EXPR.MonomialTermExpression)
        self.assertEqual(-1, temp[1].arg(1).arg(0))
        self.assertIs(m.y, temp[1].arg(1).arg(1))
        self.assertIs(type(temp[1].arg(2)), EXPR.MonomialTermExpression)
        self.assertEqual(-1, temp[1].arg(2).arg(0))
        self.assertIs(m.z, temp[1].arg(2).arg(1))
        e = m.v[t] == m.y + m.dv[t] + m.z
        temp = _check_viewsumexpression(e, 1)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.SumExpression)
        self.assertIs(type(temp[1].arg(0)), EXPR.Numeric_GetItemExpression)
        self.assertIs(type(temp[1].arg(1)), EXPR.MonomialTermExpression)
        self.assertIs(m.y, temp[1].arg(1).arg(1))
        self.assertIs(type(temp[1].arg(2)), EXPR.MonomialTermExpression)
        self.assertIs(m.z, temp[1].arg(2).arg(1))
        e = 5 * m.dv[t] + 5 * m.y - m.z == m.v[t]
        temp = _check_viewsumexpression(e, 0)
        self.assertIs(m.dv, temp[0].arg(0))
        self.assertIs(type(temp[1]), EXPR.DivisionExpression)
        self.assertIs(type(temp[1].arg(0).arg(0)), EXPR.Numeric_GetItemExpression)
        self.assertIs(m.y, temp[1].arg(0).arg(1).arg(1))
        self.assertIs(m.z, temp[1].arg(0).arg(2).arg(1))
        e = 2 + 5 * m.y - m.z == m.v[t]
        temp = _check_viewsumexpression(e, 0)
        self.assertIs(temp, None)
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core import ConcreteModel, Var, Constraint
from pyomo.gdp import Disjunction, Disjunct
from pyomo.gdp.disjunct import AutoLinkedBooleanVar, AutoLinkedBinaryVar
def test_synchronize_value(self):
    m = ConcreteModel()
    m.iv = AutoLinkedBooleanVar()
    m.biv = AutoLinkedBinaryVar(m.iv)
    m.iv.associate_binary_var(m.biv)
    self.assertIsNone(m.iv.value)
    self.assertIsNone(m.biv.value)
    m.iv = True
    self.assertEqual(m.iv.value, True)
    self.assertEqual(m.biv.value, 1)
    m.iv = True
    self.assertEqual(m.iv.value, True)
    self.assertEqual(m.biv.value, 1)
    m.iv = False
    self.assertEqual(m.iv.value, False)
    self.assertEqual(m.biv.value, 0)
    m.iv = False
    self.assertEqual(m.iv.value, False)
    self.assertEqual(m.biv.value, 0)
    m.iv = None
    self.assertEqual(m.iv.value, None)
    self.assertEqual(m.biv.value, None)
    m.iv = None
    self.assertEqual(m.iv.value, None)
    self.assertEqual(m.biv.value, None)
    m.biv = 1
    self.assertEqual(m.iv.value, True)
    self.assertEqual(m.biv.value, 1)
    m.biv = 1
    self.assertEqual(m.iv.value, True)
    self.assertEqual(m.biv.value, 1)
    eps = AutoLinkedBinaryVar.INTEGER_TOLERANCE / 10
    m.biv = None
    self.assertEqual(m.iv.value, None)
    self.assertEqual(m.biv.value, None)
    m.biv = None
    self.assertEqual(m.iv.value, None)
    self.assertEqual(m.biv.value, None)
    m.biv.value = 1 - eps
    self.assertEqual(m.iv.value, True)
    self.assertEqual(m.biv.value, 1 - eps)
    m.biv.value = 1 - eps
    self.assertEqual(m.iv.value, True)
    self.assertEqual(m.biv.value, 1 - eps)
    m.biv.value = eps
    self.assertEqual(m.iv.value, False)
    self.assertEqual(m.biv.value, eps)
    m.biv.value = eps
    self.assertEqual(m.iv.value, False)
    self.assertEqual(m.biv.value, eps)
    m.biv.value = 0.5
    self.assertEqual(m.iv.value, None)
    self.assertEqual(m.biv.value, 0.5)
    m.biv.value = 0.5
    self.assertEqual(m.iv.value, None)
    self.assertEqual(m.biv.value, 0.5)
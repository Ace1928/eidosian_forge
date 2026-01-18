from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, _ConstraintData
from pyomo.core.expr.compare import (
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import pyomo.network as ntwk
import random
from io import StringIO
def test_suffix_M_constraintKeyOnSimpleDisj(self):
    m = models.makeTwoTermDisjOnBlock()
    m = models.add_disj_not_on_block(m)
    m.simpledisj.BigM = Suffix(direction=Suffix.LOCAL)
    m.simpledisj.BigM[None] = 45
    m.simpledisj.BigM[m.simpledisj.c] = 87
    m.BigM = Suffix(direction=Suffix.LOCAL)
    m.BigM[None] = 20
    bigms = {m.b.disjunct[0].c: (-15, None)}
    bigm = TransformationFactory('gdp.bigm')
    bigm.apply_to(m, bigM=bigms)
    self.checkMs(m, -15, 20, 20, -87, 20)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.simpledisj.c)
    self.assertIs(l_src, m.simpledisj.BigM)
    self.assertIsNone(u_src)
    self.assertIs(l_key, m.simpledisj.c)
    self.assertIsNone(u_key)
    self.assertEqual(l_val, -87)
    self.assertIsNone(u_val)
    l_val, u_val = bigm.get_M_value(m.simpledisj.c)
    self.assertEqual(l_val, -87)
    self.assertIsNone(u_val)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.simpledisj2.c)
    self.assertIsNone(l_src)
    self.assertIs(u_src, m.BigM)
    self.assertIsNone(l_key)
    self.assertIsNone(u_key)
    self.assertIsNone(l_val)
    self.assertEqual(u_val, 20)
    l_val, u_val = bigm.get_M_value(m.simpledisj2.c)
    self.assertIsNone(l_val)
    self.assertEqual(u_val, 20)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.b.disjunct[0].c)
    self.assertIs(l_src, bigms)
    self.assertIs(u_src, m.BigM)
    self.assertIs(l_key, m.b.disjunct[0].c)
    self.assertIsNone(u_key)
    self.assertEqual(l_val, -15)
    self.assertEqual(u_val, 20)
    l_val, u_val = bigm.get_M_value(m.b.disjunct[0].c)
    self.assertEqual(l_val, -15)
    self.assertEqual(u_val, 20)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.b.disjunct[1].c)
    self.assertIsNone(l_src)
    self.assertIs(u_src, m.BigM)
    self.assertIsNone(l_key)
    self.assertIsNone(u_key)
    self.assertIsNone(l_val)
    self.assertEqual(u_val, 20)
    l_val, u_val = bigm.get_M_value(m.b.disjunct[1].c)
    self.assertIsNone(l_val)
    self.assertEqual(u_val, 20)
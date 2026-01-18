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
def test_suffix_M_constraintKeyOnSimpleDisj_deprecated_m_src_method(self):
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
    src, key = bigm.get_m_value_src(m.simpledisj.c)
    self.assertIs(src, m.simpledisj.BigM)
    self.assertIs(key, m.simpledisj.c)
    src, key = bigm.get_m_value_src(m.simpledisj2.c)
    self.assertIs(src, m.BigM)
    self.assertIsNone(key)
    self.assertRaisesRegex(GDP_Error, 'This is why this method is deprecated: The lower and upper M values for constraint b.disjunct\\[0\\].c came from different sources, please use the get_M_value_src method.', bigm.get_m_value_src, m.b.disjunct[0].c)
    src, key = bigm.get_m_value_src(m.b.disjunct[1].c)
    self.assertIs(src, m.BigM)
    self.assertIsNone(key)
from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
def test_virtual_indexed_constraints_in_disjunct(self):
    m = ConcreteModel()
    m.I = [1, 2, 3]
    m.x = Var(m.I, bounds=(-1, 10))

    def d_rule(d, j):
        m = d.model()
        d.c = Constraint(Any)
        for k in range(j):
            d.c[k + 1] = m.x[k + 1] >= k + 1
    m.d = Disjunct(m.I, rule=d_rule)
    m.disjunction = Disjunction(expr=[m.d[i] for i in m.I])
    TransformationFactory('gdp.hull').apply_to(m)
    self.check_threeTermDisj_IndexedConstraints(m, lb=-1)
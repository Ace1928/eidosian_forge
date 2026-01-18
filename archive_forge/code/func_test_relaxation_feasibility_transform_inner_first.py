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
@unittest.skipIf(not linear_solvers, 'No linear solver available')
def test_relaxation_feasibility_transform_inner_first(self):
    m = models.makeNestedDisjunctions_FlatDisjuncts()
    TransformationFactory('gdp.hull').apply_to(m.d1)
    TransformationFactory('gdp.hull').apply_to(m)
    solver = SolverFactory(linear_solvers[0])
    cases = [(True, True, True, True, None), (False, False, False, False, None), (True, False, False, False, None), (False, True, False, False, 1.1), (False, False, True, False, None), (False, False, False, True, None), (True, True, False, False, None), (True, False, True, False, 1.2), (True, False, False, True, 1.3), (True, False, True, True, None)]
    for case in cases:
        m.d1.indicator_var.fix(case[0])
        m.d2.indicator_var.fix(case[1])
        m.d3.indicator_var.fix(case[2])
        m.d4.indicator_var.fix(case[3])
        results = solver.solve(m)
        if case[4] is None:
            self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)
        else:
            self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
            self.assertEqual(value(m.obj), case[4])
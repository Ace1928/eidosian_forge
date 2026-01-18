import pyomo.common.unittest as unittest
import io
import logging
import math
import os
import re
import pyomo.repn.util as repn_util
import pyomo.repn.plugins.nl_writer as nl_writer
from pyomo.repn.util import InvalidNumber
from pyomo.repn.tests.nl_diff import nl_diff
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.errors import MouseTrap
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import report_timing
from pyomo.core.expr import Expr_if, inequality, LinearExpression
from pyomo.core.base.expression import ScalarExpression
from pyomo.environ import (
import pyomo.environ as pyo
def test_presolve_almost_lower_triangular(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(5), bounds=(-10, 10))
    m.obj = Objective(expr=m.x[3] + m.x[4])
    m.c = pyo.ConstraintList()
    m.c.add(m.x[0] + 2 * m.x[4] == 5)
    m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
    m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
    m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)
    OUT = io.StringIO()
    with LoggingIntercept() as LOG:
        nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
    self.assertEqual(LOG.getvalue(), '')
    self.assertIs(nlinfo.eliminated_vars[0][0], m.x[4])
    self.assertExpressionsEqual(nlinfo.eliminated_vars[0][1], 3.0 * m.x[1] - 12.0)
    self.assertIs(nlinfo.eliminated_vars[1][0], m.x[3])
    self.assertExpressionsEqual(nlinfo.eliminated_vars[1][1], 17.0 * m.x[1] - 72.0)
    self.assertIs(nlinfo.eliminated_vars[2][0], m.x[2])
    self.assertExpressionsEqual(nlinfo.eliminated_vars[2][1], 4.0 * m.x[1] - 13.0)
    self.assertIs(nlinfo.eliminated_vars[3][0], m.x[0])
    self.assertExpressionsEqual(nlinfo.eliminated_vars[3][1], -6.0 * m.x[1] + 29.0)
    self.assertEqual(*nl_diff('g3 1 1 0\t# problem unknown\n 1 0 1 0 0 \t# vars, constraints, objectives, ranges, eqns\n 0 0 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 0 0 0 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 0 1 \t# nonzeros in Jacobian, obj. gradient\n 0 0\t# max name lengths: constraints, variables\n 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\nO0 0\nn-84.0\nx0\nr\nb\n0 3.6470588235294117 4.823529411764706\nk0\nG0 1\n0 20\n', OUT.getvalue()))
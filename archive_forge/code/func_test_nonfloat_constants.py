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
@unittest.skipUnless(numpy_available, 'test requires numpy')
def test_nonfloat_constants(self):
    import pyomo.environ as pyo
    v = numpy.array([[8], [3], [6], [11]])
    w = numpy.array([[5], [7], [4], [3]])
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=range(4))
    m.zero = pyo.Param(initialize=numpy.array([0]), mutable=True)
    m.one = pyo.Param(initialize=numpy.array([1]), mutable=True)
    m.x = pyo.Var(m.I, bounds=(m.zero, m.one), domain=pyo.Integers, initialize=True)
    m.limit = pyo.Param(initialize=numpy.array([14]), mutable=True)
    m.v = pyo.Param(m.I, initialize=v, mutable=True)
    m.w = pyo.Param(m.I, initialize=w, mutable=True)
    m.value = pyo.Objective(expr=pyo.sum_product(m.v, m.x), sense=pyo.maximize)
    m.weight = pyo.Constraint(expr=pyo.sum_product(m.w, m.x) <= m.limit)
    OUT = io.StringIO()
    ROW = io.StringIO()
    COL = io.StringIO()
    with LoggingIntercept() as LOG:
        nl_writer.NLWriter().write(m, OUT, ROW, COL, symbolic_solver_labels=True)
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(ROW.getvalue(), 'weight\nvalue\n')
    self.assertEqual(COL.getvalue(), 'x[0]\nx[1]\nx[2]\nx[3]\n')
    self.assertEqual(*nl_diff("g3 1 1 0       #problem unknown\n 4 1 1 0 0     #vars, constraints, objectives, ranges, eqns\n 0 0 0 0 0 0   #nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0   #network constraints: nonlinear, linear\n 0 0 0 #nonlinear vars in constraints, objectives, both\n 0 0 0 1       #linear network variables; functions; arith, flags\n 0 4 0 0 0     #discrete variables: binary, integer, nonlinear (b,c,o)\n 4 4   #nonzeros in Jacobian, obj. gradient\n 6 4   #max name lengths: constraints, variables\n 0 0 0 0 0     #common exprs: b,c,o,c1,o1\nC0     #weight\nn0\nO0 1   #value\nn0\nx4     #initial guess\n0 1.0  #x[0]\n1 1.0  #x[1]\n2 1.0  #x[2]\n3 1.0  #x[3]\nr      #1 ranges (rhs's)\n1 14.0 #weight\nb      #4 bounds (on variables)\n0 0 1  #x[0]\n0 0 1  #x[1]\n0 0 1  #x[2]\n0 0 1  #x[3]\nk3     #intermediate Jacobian column lengths\n1\n2\n3\nJ0 4   #weight\n0 5\n1 7\n2 4\n3 3\nG0 4   #value\n0 8\n1 3\n2 6\n3 11\n", OUT.getvalue()))
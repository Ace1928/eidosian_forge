import pyomo.environ as pe
from pyomo.util.report_scaling import report_scaling
import logging
from pyomo.common import unittest
from pyomo.common.log import LoggingIntercept
from io import StringIO
import re
def test_report_scaling(self):
    m = pe.ConcreteModel()
    m.x = pe.Var(list(range(5)))
    m.c = pe.Constraint(list(range(4)))
    m.p = pe.Param(initialize=0, mutable=True)
    m.x[0].setlb(-1)
    m.x[0].setub(1)
    m.x[1].setlb(100000.0)
    m.x[1].setub(10000000.0)
    m.x[2].setlb(-100000.0)
    m.x[2].setub(0)
    m.x[3].setlb(-20)
    m.x[3].setub(20)
    m.obj1 = pe.Objective(expr=1e-08 * m.x[0] + pe.exp(m.x[3]) + m.x[1] * m.x[2])
    m.obj2 = pe.Objective(expr=m.x[0] * m.x[3] + m.x[1] ** 2)
    m.c[0] = m.x[0] + m.x[3] == 0
    m.c[1] = 1 / m.x[1] == 1
    m.c[2] = m.x[1] * m.x[3] == 1
    m.c[3] = m.x[3] + m.p * m.x[0] == 1
    out = StringIO()
    with LoggingIntercept(out, 'pyomo.util.report_scaling', level=logging.INFO):
        report_scaling(m)
    expected = '\n\nThe following variables are not bounded. Please add bounds.\n          LB          UB    Var\n        -inf         inf    x[4]\n\nThe following variables have large bounds. Please scale them.\n          LB          UB    Var\n    1.00e+05    1.00e+07    x[1]\n   -1.00e+05    0.00e+00    x[2]\n\nThe following objectives have potentially large coefficients. Please scale them.\nobj1\n         Coef LB     Coef UB    Var\n        2.06e-09    4.85e+08    x[3]\n       -1.00e+05    0.00e+00    x[1]\n        1.00e+05    1.00e+07    x[2]\n\nobj2\n         Coef LB     Coef UB    Var\n        2.00e+05    2.00e+07    x[1]\n\nThe following objectives have small coefficients.\nobj1\n         Coef LB     Coef UB    Var\n        1.00e-08    1.00e-08    x[0]\n\nThe following constraints have potentially large coefficients. Please scale them.\nc[2]\n         Coef LB     Coef UB    Var\n        1.00e+05    1.00e+07    x[3]\n\nThe following constraints have small coefficients.\nc[1]\n         Coef LB     Coef UB    Var\n       -1.00e-10   -1.00e-14    x[1]\n\nThe following constraints have bodies with large bounds. Please scale them.\n          LB          UB    Constraint\n   -2.00e+08    2.00e+08    c[2]\n\n'
    self.assertEqual(out.getvalue(), expected)
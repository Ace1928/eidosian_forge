import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.contrib.iis.iis import _supported_solvers
from pyomo.common.tempfiles import TempfileManager
import os
@unittest.skipUnless(pyo.SolverFactory('gurobi_persistent').available(exception_flag=False), 'Gurobi not available')
def test_write_iis_gurobi(self):
    _test_iis('gurobi')
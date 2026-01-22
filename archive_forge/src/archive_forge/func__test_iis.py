import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.contrib.iis.iis import _supported_solvers
from pyomo.common.tempfiles import TempfileManager
import os
def _test_iis(solver_name):
    m = _get_infeasible_model()
    TempfileManager.push()
    tmp_path = TempfileManager.create_tempdir()
    file_name = os.path.join(tmp_path, f'{solver_name}_iis.ilp')
    file_name = write_iis(m, solver=solver_name, iis_file_name=str(file_name))
    _validate_ilp(file_name)
    TempfileManager.pop()
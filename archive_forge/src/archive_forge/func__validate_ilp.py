import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.contrib.iis.iis import _supported_solvers
from pyomo.common.tempfiles import TempfileManager
import os
def _validate_ilp(file_name):
    lines_found = {'c2: 100 x + y <= 0': False, 'c3: x >= 0.5': False}
    with open(file_name, 'r') as f:
        for line in f.readlines():
            for k, v in lines_found.items():
                if not v and k in line:
                    lines_found[k] = True
    if not all(lines_found.values()):
        raise Exception(f'The file {file_name} is not as expected. Missing constraints:\n' + '\n'.join((k for k, v in lines_found.items() if not v)))
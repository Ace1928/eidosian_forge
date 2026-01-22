import logging
import os
import subprocess
import re
import tempfile
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import SystemCallSolver
def _get_dummy_input_files(self, check_license=False):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as fr:
            pass
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as fs:
            pass
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as ft:
            pass
        f.write('//This is a dummy .bar file created to return the baron version//\nOPTIONS {\nresults: 1;\nResName: "' + fr.name + '";\nsummary: 1;\nSumName: "' + fs.name + '";\ntimes: 1;\nTimName: "' + ft.name + '";\n}\n')
        f.write('POSITIVE_VARIABLES ')
        if check_license:
            f.write(', '.join(('x' + str(i) for i in range(11))))
        else:
            f.write('x1')
        f.write(';\n')
        f.write('OBJ: minimize x1;')
    return (f.name, fr.name, fs.name, ft.name)
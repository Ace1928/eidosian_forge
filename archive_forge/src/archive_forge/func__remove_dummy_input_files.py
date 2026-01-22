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
def _remove_dummy_input_files(self, fnames):
    for name in fnames:
        try:
            os.remove(name)
        except OSError:
            pass
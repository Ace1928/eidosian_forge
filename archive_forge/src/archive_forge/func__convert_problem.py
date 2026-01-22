import os
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.kernel.block import IBlock
from pyomo.core import Var
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import SystemCallSolver
from pyomo.solvers.mockmip import MockMIP
def _convert_problem(self, args, pformat, valid_pformats):
    if pformat in [ProblemFormat.mps, ProblemFormat.cpxlp, ProblemFormat.nl]:
        return (args, pformat, None)
    else:
        return (args, ProblemFormat.mps, None)
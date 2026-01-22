import os
import subprocess
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat, ResultsFormat
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import SolverStatus
from pyomo.opt.solver import SystemCallSolver
import logging

        Returns a tuple describing the solver executable version.
        
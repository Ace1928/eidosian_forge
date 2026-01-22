import logging
from pyomo.opt.base import SolverFactory, ProblemFormat, ResultsFormat
from pyomo.opt.solver import SystemCallSolver
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager

        Create the local *.sol and *.log files, which will be
        populated by NEOS.
        
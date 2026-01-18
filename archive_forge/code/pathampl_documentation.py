import logging
from pyomo.opt.base.solvers import SolverFactory
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.solvers.plugins.solvers.ASL import ASL
An interface to the PATH MCP solver.
import os
import pyomo.common.unittest as unittest
from pyomo.opt import (
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.plugins.sol import ResultsReader_sol
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
class MockWriter(AbstractProblemWriter):

    def __init__(self, name=None):
        AbstractProblemWriter.__init__(self, name)
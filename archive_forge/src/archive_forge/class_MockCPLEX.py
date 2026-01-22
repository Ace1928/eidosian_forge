import os
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections import ComponentMap, Bunch
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver, BranchDirection
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import ILMLicensedSystemCallSolver
from pyomo.solvers.mockmip import MockMIP
from pyomo.core.base import Var, Suffix, active_export_suffix_generator
from pyomo.core.kernel.suffix import export_suffix_generator
from pyomo.core.kernel.block import IBlock
from pyomo.util.components import iter_component
@SolverFactory.register('_mock_cplex')
class MockCPLEX(CPLEXSHELL, MockMIP):
    """A Mock CPLEX solver used for testing"""

    def __init__(self, **kwds):
        try:
            CPLEXSHELL.__init__(self, **kwds)
        except ApplicationError:
            pass
        MockMIP.__init__(self, 'cplex')

    def available(self, exception_flag=True):
        return CPLEXSHELL.available(self, exception_flag)

    def create_command_line(self, executable, problem_files):
        command = CPLEXSHELL.create_command_line(self, executable, problem_files)
        MockMIP.create_command_line(self, executable, problem_files)
        return command

    def _default_executable(self):
        return MockMIP.executable(self)

    def _execute_command(self, cmd):
        return MockMIP._execute_command(self, cmd)
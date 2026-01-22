import os
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat, ResultsFormat
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.solver import SystemCallSolver
from pyomo.core.kernel.block import IBlock
from pyomo.solvers.mockmip import MockMIP
from pyomo.core import TransformationFactory
import logging
@SolverFactory.register('_mock_asl')
class MockASL(ASL, MockMIP):
    """A Mock ASL solver used for testing"""

    def __init__(self, **kwds):
        try:
            ASL.__init__(self, **kwds)
        except ApplicationError:
            pass
        MockMIP.__init__(self, 'asl')
        self._assert_available = True

    def available(self, exception_flag=True):
        return ASL.available(self, exception_flag)

    def create_command_line(self, executable, problem_files):
        command = ASL.create_command_line(self, executable, problem_files)
        MockMIP.create_command_line(self, executable, problem_files)
        return command

    def executable(self):
        return MockMIP.executable(self)

    def _execute_command(self, cmd):
        return MockMIP._execute_command(self, cmd)
import os.path
from pyomo.common.fileutils import this_file_dir, import_file
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
from pyomo.opt import TerminationCondition
from io import StringIO
import logging
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver as cyipopt_solver
import cyipopt as cyipopt_core
class TestPyomoCyIpoptSolver(unittest.TestCase):

    def test_status_maps(self):
        for msg in cyipopt_core.STATUS_MESSAGES.values():
            self.assertIn(msg, cyipopt_solver._cyipopt_status_enum)
        for status in cyipopt_solver._cyipopt_status_enum.values():
            self.assertIn(status, cyipopt_solver._ipopt_term_cond)
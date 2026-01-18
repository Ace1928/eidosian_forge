import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
def test_raise_due_to_invalid_priority(self):
    self.mock_model.priority = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
    self._set_suffix_value(self.mock_model.priority, self.mock_model.x, -1)
    with self.assertRaises(ValueError):
        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
    self._set_suffix_value(self.mock_model.priority, self.mock_model.x, 1.1)
    with self.assertRaises(ValueError):
        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
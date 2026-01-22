import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
class TestImplicitFunctionSolver(_TestSolver):

    def get_solver_class(self):
        return ImplicitFunctionSolver

    def test_bad_option(self):
        msg = 'Option.*is invalid'
        with self.assertRaisesRegex(ValueError, msg):
            self._test_implicit_function_1(solver_options=dict(bad_option=None))

    def test_implicit_function_1(self):
        self._test_implicit_function_1()

    @unittest.skipUnless(cyipopt_available, 'CyIpopt is not available')
    def test_implicit_function_1_with_cyipopt(self):
        self._test_implicit_function_1(solver_class=CyIpoptSolverWrapper)

    def test_implicit_function_inputs_dont_appear(self):
        self._test_implicit_function_inputs_dont_appear()

    def test_implicit_function_no_inputs(self):
        self._test_implicit_function_no_inputs()

    def test_implicit_function_with_extra_variables(self):
        self._test_implicit_function_with_extra_variables()
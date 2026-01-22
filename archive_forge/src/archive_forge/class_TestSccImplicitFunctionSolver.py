import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
@unittest.skipUnless(networkx_available, 'NetworkX is not available')
class TestSccImplicitFunctionSolver(_TestSolver):

    def get_solver_class(self):
        return SccImplicitFunctionSolver

    def test_partition_not_implemented(self):
        fcn = ImplicitFunction1()
        variables = fcn.get_variables()
        parameters = fcn.get_parameters()
        equations = fcn.get_equations()
        msg = 'has not implemented'
        with self.assertRaisesRegex(NotImplementedError, msg):
            solver = DecomposedImplicitFunctionBase(variables, equations, parameters)

    def test_n_subsystems(self):
        SolverClass = self.get_solver_class()
        fcn = ImplicitFunction1()
        variables = fcn.get_variables()
        parameters = fcn.get_parameters()
        equations = fcn.get_equations()
        solver = SolverClass(variables, equations, parameters)
        self.assertEqual(solver.n_subsystems(), 2)

    def test_implicit_function_1(self):
        self._test_implicit_function_1()

    @unittest.skipUnless(cyipopt_available, 'CyIpopt is not available')
    def test_implicit_function_1_with_cyipopt(self):
        self._test_implicit_function_1(solver_class=CyIpoptSolverWrapper)

    def test_implicit_function_1_no_calc_var(self):
        self._test_implicit_function_1(use_calc_var=False, solver_options={'maxfev': 20})

    def test_implicit_function_inputs_dont_appear(self):
        self._test_implicit_function_inputs_dont_appear()

    def test_implicit_function_no_inputs(self):
        self._test_implicit_function_no_inputs()

    def test_implicit_function_with_extra_variables(self):
        self._test_implicit_function_with_extra_variables()
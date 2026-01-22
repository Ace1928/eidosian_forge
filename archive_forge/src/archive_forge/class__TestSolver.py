import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
class _TestSolver(unittest.TestCase):
    """A suite of basic tests for implicit function solvers.

    A "concrete" subclass should be defined for each implicit function
    solver. This subclass should implement get_solver_class, then
    add "test" methods that call the following methods:

        _test_implicit_function_1
        _test_implicit_function_inputs_dont_appear
        _test_implicit_function_no_inputs
        _test_implicit_function_with_extra_variables

    These methods are private so they don't get picked up on the base
    class by pytest.

    """

    def get_solver_class(self):
        raise NotImplementedError()

    def _test_implicit_function(self, ImplicitFunctionClass, **kwds):
        SolverClass = self.get_solver_class()
        fcn = ImplicitFunctionClass()
        variables = fcn.get_variables()
        parameters = fcn.get_parameters()
        equations = fcn.get_equations()
        solver = SolverClass(variables, equations, parameters, **kwds)
        for inputs, pred_outputs in fcn.get_input_output_sequence():
            solver.set_parameters(inputs)
            outputs = solver.evaluate_outputs()
            self.assertStructuredAlmostEqual(list(outputs), list(pred_outputs), reltol=1e-05, abstol=1e-05)
            solver.update_pyomo_model()
            for i, var in enumerate(variables):
                self.assertAlmostEqual(var.value, pred_outputs[i], delta=1e-05)

    def _test_implicit_function_1(self, **kwds):
        self._test_implicit_function(ImplicitFunction1, **kwds)

    def _test_implicit_function_inputs_dont_appear(self):
        self._test_implicit_function(ImplicitFunctionInputsDontAppear)

    def _test_implicit_function_no_inputs(self):
        self._test_implicit_function(ImplicitFunctionNoInputs)

    def _test_implicit_function_with_extra_variables(self):
        self._test_implicit_function(ImplicitFunctionWithExtraVariables)
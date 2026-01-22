import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
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
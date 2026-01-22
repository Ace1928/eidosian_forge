from sys import version_info as _swig_python_version_info
import numbers
from ortools.linear_solver.python.linear_solver_natural_api import OFFSET_KEY
from ortools.linear_solver.python.linear_solver_natural_api import inf
from ortools.linear_solver.python.linear_solver_natural_api import LinearExpr
from ortools.linear_solver.python.linear_solver_natural_api import ProductCst
from ortools.linear_solver.python.linear_solver_natural_api import Sum
from ortools.linear_solver.python.linear_solver_natural_api import SumArray
from ortools.linear_solver.python.linear_solver_natural_api import SumCst
from ortools.linear_solver.python.linear_solver_natural_api import LinearConstraint
from ortools.linear_solver.python.linear_solver_natural_api import VariableExpr
def Maximize(self, expr):
    objective = self.Objective()
    objective.Clear()
    objective.SetMaximization()
    if isinstance(expr, numbers.Number):
        objective.SetOffset(expr)
    else:
        coeffs = expr.GetCoeffs()
        objective.SetOffset(coeffs.pop(OFFSET_KEY, 0.0))
        for v, c in list(coeffs.items()):
            objective.SetCoefficient(v, float(c))
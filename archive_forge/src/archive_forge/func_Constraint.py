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
def Constraint(self, *args):
    """
        *Overload 1:*

        Creates a linear constraint with given bounds.

        Bounds can be finite or +/- MPSolver::infinity(). The MPSolver class
        assumes ownership of the constraint.

        :rtype: :py:class:`MPConstraint`
        :return: a pointer to the newly created constraint.

        |

        *Overload 2:*
         Creates a constraint with -infinity and +infinity bounds.

        |

        *Overload 3:*
         Creates a named constraint with given bounds.

        |

        *Overload 4:*
         Creates a named constraint with -infinity and +infinity bounds.
        """
    return _pywraplp.Solver_Constraint(self, *args)
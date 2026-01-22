from cvxpy.expressions import cvxtypes
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.reductions.reduction import Reduction
Map the solution of the flipped problem to that of the original.

        Parameters
        ----------
        solution : Solution
            A solution object.
        inverse_data : list
            The inverse data returned by an invocation to apply.

        Returns
        -------
        Solution
            A solution to the original problem.
        
from typing import List, Tuple
import numpy as np
import cvxpy.utilities.performance_utils as perf
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.
        None indicates variable values unknown or outside domain.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        
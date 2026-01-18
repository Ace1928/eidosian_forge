import numpy as np
from scipy import linalg as LA
from cvxpy.atoms.lambda_max import lambda_max
from cvxpy.atoms.sum_largest import sum_largest
Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        
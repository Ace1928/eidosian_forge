from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.norm import norm
from cvxpy.expressions.expression import Expression
Total variation of a vector, matrix, or list of matrices.

    Uses L1 norm of discrete gradients for vectors and
    L2 norm of discrete gradients for matrices.

    Parameters
    ----------
    value : Expression or numeric constant
        The value to take the total variation of.
    args : Matrix constants/expressions
        Additional matrices extending the third dimension of value.

    Returns
    -------
    Expression
        An Expression representing the total variation.
    
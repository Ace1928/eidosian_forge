from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.expression import Expression
Flattens the matrix X into a vector.

    Parameters
    ----------
    X : Expression or numeric constant
        The matrix to flatten.
    order: column-major ('F') or row-major ('C') order.

    Returns
    -------
    Expression
        An Expression representing the flattened matrix.
    
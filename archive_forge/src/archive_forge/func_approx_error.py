import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def approx_error(a_orig, w_approx):
    """ Return the :math:`\\ell_\\infty` norm error from approximating the vector a_orig/sum(a_orig)
        with the weight vector w_approx.

        That is, return

        .. math:: \\|a/\\mathbf{1}^T a - w_{\\mbox{approx}} \\|_\\infty


        >>> e = approx_error([1, 1, 1], [Fraction(1,3), Fraction(1,3), Fraction(1,3)])
        >>> e <= 1e-10
        True
    """
    assert all((v >= 0 for v in a_orig))
    assert is_weight(w_approx)
    assert len(a_orig) == len(w_approx)
    w_orig = np.array(a_orig, dtype=float) / sum(a_orig)
    return float(max((abs(v1 - v2) for v1, v2 in zip(w_orig, w_approx))))
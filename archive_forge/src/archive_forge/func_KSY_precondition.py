from math import prod
from sympy.core.mul import Mul
from sympy.matrices.dense import (Matrix, diag)
from sympy.polys.polytools import (Poly, degree_list, rem)
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import IndexedBase
from sympy.polys.monomials import itermonomials, monomial_deg
from sympy.polys.orderings import monomial_key
from sympy.polys.polytools import poly_from_expr, total_degree
from sympy.functions.combinatorial.factorials import binomial
from itertools import combinations_with_replacement
from sympy.utilities.exceptions import sympy_deprecation_warning
def KSY_precondition(self, matrix):
    """
        Test for the validity of the Kapur-Saxena-Yang precondition.

        The precondition requires that the column corresponding to the
        monomial 1 = x_1 ^ 0 * x_2 ^ 0 * ... * x_n ^ 0 is not a linear
        combination of the remaining ones. In SymPy notation this is
        the last column. For the precondition to hold the last non-zero
        row of the rref matrix should be of the form [0, 0, ..., 1].
        """
    if matrix.is_zero_matrix:
        return False
    m, n = matrix.shape
    matrix = simplify(matrix.rref()[0])
    rows = [i for i in range(m) if any((matrix[i, j] != 0 for j in range(n)))]
    matrix = matrix[rows, :]
    condition = Matrix([[0] * (n - 1) + [1]])
    if matrix[-1, :] == condition:
        return True
    else:
        return False
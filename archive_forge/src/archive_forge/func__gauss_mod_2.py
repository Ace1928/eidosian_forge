from sympy.core.numbers import igcd, mod_inverse
from sympy.core.power import integer_nthroot
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt
import random
def _gauss_mod_2(A):
    """Fast gaussian reduction for modulo 2 matrix.

    Parameters
    ==========

    A : Matrix

    Examples
    ========

    >>> from sympy.ntheory.qs import _gauss_mod_2
    >>> _gauss_mod_2([[0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1]])
    ([[[1, 0, 1], 3]],
     [True, True, True, False],
     [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]])

    Reference
    ==========

    .. [1] A fast algorithm for gaussian elimination over GF(2) and
    its implementation on the GAPP. Cetin K.Koc, Sarath N.Arachchige"""
    import copy
    matrix = copy.deepcopy(A)
    row = len(matrix)
    col = len(matrix[0])
    mark = [False] * row
    for c in range(col):
        for r in range(row):
            if matrix[r][c] == 1:
                break
        mark[r] = True
        for c1 in range(col):
            if c1 == c:
                continue
            if matrix[r][c1] == 1:
                for r2 in range(row):
                    matrix[r2][c1] = (matrix[r2][c1] + matrix[r2][c]) % 2
    dependent_row = []
    for idx, val in enumerate(mark):
        if val == False:
            dependent_row.append([matrix[idx], idx])
    return (dependent_row, mark, matrix)
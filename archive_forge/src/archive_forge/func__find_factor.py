from sympy.core.numbers import igcd, mod_inverse
from sympy.core.power import integer_nthroot
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt
import random
def _find_factor(dependent_rows, mark, gauss_matrix, index, smooth_relations, N):
    """Finds proper factor of N. Here, transform the dependent rows as a
    combination of independent rows of the gauss_matrix to form the desired
    relation of the form ``X**2 = Y**2 modN``. After obtaining the desired relation
    we obtain a proper factor of N by `gcd(X - Y, N)`.

    Parameters
    ==========

    dependent_rows : denoted dependent rows in the reduced matrix form
    mark : boolean array to denoted dependent and independent rows
    gauss_matrix : Reduced form of the smooth relations matrix
    index : denoted the index of the dependent_rows
    smooth_relations : Smooth relations vectors matrix
    N : Number to be factored
    """
    idx_in_smooth = dependent_rows[index][1]
    independent_u = [smooth_relations[idx_in_smooth][0]]
    independent_v = [smooth_relations[idx_in_smooth][1]]
    dept_row = dependent_rows[index][0]
    for idx, val in enumerate(dept_row):
        if val == 1:
            for row in range(len(gauss_matrix)):
                if gauss_matrix[row][idx] == 1 and mark[row] == True:
                    independent_u.append(smooth_relations[row][0])
                    independent_v.append(smooth_relations[row][1])
                    break
    u = 1
    v = 1
    for i in independent_u:
        u *= i
    for i in independent_v:
        v *= i
    v = integer_nthroot(v, 2)[0]
    return igcd(u - v, N)
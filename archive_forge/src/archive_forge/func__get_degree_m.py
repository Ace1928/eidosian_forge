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
def _get_degree_m(self):
    """
        Returns
        =======

        degree_m: int
            The degree_m is calculated as  1 + \\sum_1 ^ n (d_i - 1),
            where d_i is the degree of the i polynomial
        """
    return 1 + sum((d - 1 for d in self.degrees))
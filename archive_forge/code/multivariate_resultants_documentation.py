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

        Returns
        =======

        macaulay_submatrix: Matrix
            The Macaulay denominator matrix. Columns that are non reduced are kept.
            The row which contains one of the a_{i}s is dropped. a_{i}s
            are the coefficients of x_i ^ {d_i}.
        
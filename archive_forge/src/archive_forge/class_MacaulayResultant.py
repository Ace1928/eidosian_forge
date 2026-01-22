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
class MacaulayResultant:
    """
    A class for calculating the Macaulay resultant. Note that the
    polynomials must be homogenized and their coefficients must be
    given as symbols.

    Examples
    ========

    >>> from sympy import symbols

    >>> from sympy.polys.multivariate_resultants import MacaulayResultant
    >>> x, y, z = symbols('x, y, z')

    >>> a_0, a_1, a_2 = symbols('a_0, a_1, a_2')
    >>> b_0, b_1, b_2 = symbols('b_0, b_1, b_2')
    >>> c_0, c_1, c_2,c_3, c_4 = symbols('c_0, c_1, c_2, c_3, c_4')

    >>> f = a_0 * y -  a_1 * x + a_2 * z
    >>> g = b_1 * x ** 2 + b_0 * y ** 2 - b_2 * z ** 2
    >>> h = c_0 * y * z ** 2 - c_1 * x ** 3 + c_2 * x ** 2 * z - c_3 * x * z ** 2 + c_4 * z ** 3

    >>> mac = MacaulayResultant(polynomials=[f, g, h], variables=[x, y, z])
    >>> mac.monomial_set
    [x**4, x**3*y, x**3*z, x**2*y**2, x**2*y*z, x**2*z**2, x*y**3,
    x*y**2*z, x*y*z**2, x*z**3, y**4, y**3*z, y**2*z**2, y*z**3, z**4]
    >>> matrix = mac.get_matrix()
    >>> submatrix = mac.get_submatrix(matrix)
    >>> submatrix
    Matrix([
    [-a_1,  a_0,  a_2,    0],
    [   0, -a_1,    0,    0],
    [   0,    0, -a_1,    0],
    [   0,    0,    0, -a_1]])

    See Also
    ========

    Notebook in examples: sympy/example/notebooks.

    References
    ==========

    .. [1] [Bruce97]_
    .. [2] [Stiller96]_

    """

    def __init__(self, polynomials, variables):
        """
        Parameters
        ==========

        variables: list
            A list of all n variables
        polynomials : list of SymPy polynomials
            A  list of m n-degree polynomials
        """
        self.polynomials = polynomials
        self.variables = variables
        self.n = len(variables)
        self.degrees = [total_degree(poly, *self.variables) for poly in self.polynomials]
        self.degree_m = self._get_degree_m()
        self.monomials_size = self.get_size()
        self.monomial_set = self.get_monomials_of_certain_degree(self.degree_m)

    def _get_degree_m(self):
        """
        Returns
        =======

        degree_m: int
            The degree_m is calculated as  1 + \\sum_1 ^ n (d_i - 1),
            where d_i is the degree of the i polynomial
        """
        return 1 + sum((d - 1 for d in self.degrees))

    def get_size(self):
        """
        Returns
        =======

        size: int
            The size of set T. Set T is the set of all possible
            monomials of the n variables for degree equal to the
            degree_m
        """
        return binomial(self.degree_m + self.n - 1, self.n - 1)

    def get_monomials_of_certain_degree(self, degree):
        """
        Returns
        =======

        monomials: list
            A list of monomials of a certain degree.
        """
        monomials = [Mul(*monomial) for monomial in combinations_with_replacement(self.variables, degree)]
        return sorted(monomials, reverse=True, key=monomial_key('lex', self.variables))

    def get_row_coefficients(self):
        """
        Returns
        =======

        row_coefficients: list
            The row coefficients of Macaulay's matrix
        """
        row_coefficients = []
        divisible = []
        for i in range(self.n):
            if i == 0:
                degree = self.degree_m - self.degrees[i]
                monomial = self.get_monomials_of_certain_degree(degree)
                row_coefficients.append(monomial)
            else:
                divisible.append(self.variables[i - 1] ** self.degrees[i - 1])
                degree = self.degree_m - self.degrees[i]
                poss_rows = self.get_monomials_of_certain_degree(degree)
                for div in divisible:
                    for p in poss_rows:
                        if rem(p, div) == 0:
                            poss_rows = [item for item in poss_rows if item != p]
                row_coefficients.append(poss_rows)
        return row_coefficients

    def get_matrix(self):
        """
        Returns
        =======

        macaulay_matrix: Matrix
            The Macaulay numerator matrix
        """
        rows = []
        row_coefficients = self.get_row_coefficients()
        for i in range(self.n):
            for multiplier in row_coefficients[i]:
                coefficients = []
                poly = Poly(self.polynomials[i] * multiplier, *self.variables)
                for mono in self.monomial_set:
                    coefficients.append(poly.coeff_monomial(mono))
                rows.append(coefficients)
        macaulay_matrix = Matrix(rows)
        return macaulay_matrix

    def get_reduced_nonreduced(self):
        """
        Returns
        =======

        reduced: list
            A list of the reduced monomials
        non_reduced: list
            A list of the monomials that are not reduced

        Definition
        ==========

        A polynomial is said to be reduced in x_i, if its degree (the
        maximum degree of its monomials) in x_i is less than d_i. A
        polynomial that is reduced in all variables but one is said
        simply to be reduced.
        """
        divisible = []
        for m in self.monomial_set:
            temp = []
            for i, v in enumerate(self.variables):
                temp.append(bool(total_degree(m, v) >= self.degrees[i]))
            divisible.append(temp)
        reduced = [i for i, r in enumerate(divisible) if sum(r) < self.n - 1]
        non_reduced = [i for i, r in enumerate(divisible) if sum(r) >= self.n - 1]
        return (reduced, non_reduced)

    def get_submatrix(self, matrix):
        """
        Returns
        =======

        macaulay_submatrix: Matrix
            The Macaulay denominator matrix. Columns that are non reduced are kept.
            The row which contains one of the a_{i}s is dropped. a_{i}s
            are the coefficients of x_i ^ {d_i}.
        """
        reduced, non_reduced = self.get_reduced_nonreduced()
        if reduced == []:
            return diag([1])
        reduction_set = [v ** self.degrees[i] for i, v in enumerate(self.variables)]
        ais = [self.polynomials[i].coeff(reduction_set[i]) for i in range(self.n)]
        reduced_matrix = matrix[:, reduced]
        keep = []
        for row in range(reduced_matrix.rows):
            check = [ai in reduced_matrix[row, :] for ai in ais]
            if True not in check:
                keep.append(row)
        return matrix[keep, non_reduced]
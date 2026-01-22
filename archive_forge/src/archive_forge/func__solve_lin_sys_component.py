from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import connected_components
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer, Rational
from sympy.matrices.dense import MutableDenseMatrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.domains import EX
from sympy.polys.rings import sring
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.domainmatrix import DomainMatrix
def _solve_lin_sys_component(eqs_coeffs, eqs_rhs, ring):
    """Solve a linear system from dict of PolynomialRing coefficients

    Explanation
    ===========

    This is an **internal** function used by :func:`solve_lin_sys` after the
    equations have been preprocessed. After :func:`_solve_lin_sys` splits the
    system into connected components this function is called for each
    component. The system of equations is solved using Gauss-Jordan
    elimination with division followed by back-substitution.

    Examples
    ========

    Setup a system for $x-y=0$ and $x+y=2$ and solve:

    >>> from sympy import symbols, sring
    >>> from sympy.polys.solvers import _solve_lin_sys_component
    >>> x, y = symbols('x, y')
    >>> R, (xr, yr) = sring([x, y], [x, y])
    >>> eqs = [{xr:R.one, yr:-R.one}, {xr:R.one, yr:R.one}]
    >>> eqs_rhs = [R.zero, -2*R.one]
    >>> _solve_lin_sys_component(eqs, eqs_rhs, R)
    {y: 1, x: 1}

    See also
    ========

    solve_lin_sys: This function is used internally by :func:`solve_lin_sys`.
    """
    matrix = eqs_to_matrix(eqs_coeffs, eqs_rhs, ring.gens, ring.domain)
    if not matrix.domain.is_Field:
        matrix = matrix.to_field()
    echelon, pivots = matrix.rref()
    keys = ring.gens
    if pivots and pivots[-1] == len(keys):
        return None
    if len(pivots) == len(keys):
        sol = []
        for s in [row[-1] for row in echelon.rep.to_ddm()]:
            a = s
            sol.append(a)
        sols = dict(zip(keys, sol))
    else:
        sols = {}
        g = ring.gens
        if hasattr(ring, 'ring'):
            convert = ring.ring.ground_new
        else:
            convert = ring.ground_new
        echelon = echelon.rep.to_ddm()
        vals_set = {v for row in echelon for v in row}
        vals_map = {v: convert(v) for v in vals_set}
        echelon = [[vals_map[eij] for eij in ei] for ei in echelon]
        for i, p in enumerate(pivots):
            v = echelon[i][-1] - sum((echelon[i][j] * g[j] for j in range(p + 1, len(g)) if echelon[i][j]))
            sols[keys[p]] = v
    return sols
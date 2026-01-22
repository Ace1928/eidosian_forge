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
def _solve_lin_sys(eqs_coeffs, eqs_rhs, ring):
    """Solve a linear system from dict of PolynomialRing coefficients

    Explanation
    ===========

    This is an **internal** function used by :func:`solve_lin_sys` after the
    equations have been preprocessed. The role of this function is to split
    the system into connected components and pass those to
    :func:`_solve_lin_sys_component`.

    Examples
    ========

    Setup a system for $x-y=0$ and $x+y=2$ and solve:

    >>> from sympy import symbols, sring
    >>> from sympy.polys.solvers import _solve_lin_sys
    >>> x, y = symbols('x, y')
    >>> R, (xr, yr) = sring([x, y], [x, y])
    >>> eqs = [{xr:R.one, yr:-R.one}, {xr:R.one, yr:R.one}]
    >>> eqs_rhs = [R.zero, -2*R.one]
    >>> _solve_lin_sys(eqs, eqs_rhs, R)
    {y: 1, x: 1}

    See also
    ========

    solve_lin_sys: This function is used internally by :func:`solve_lin_sys`.
    """
    V = ring.gens
    E = []
    for eq_coeffs in eqs_coeffs:
        syms = list(eq_coeffs)
        E.extend(zip(syms[:-1], syms[1:]))
    G = (V, E)
    components = connected_components(G)
    sym2comp = {}
    for n, component in enumerate(components):
        for sym in component:
            sym2comp[sym] = n
    subsystems = [([], []) for _ in range(len(components))]
    for eq_coeff, eq_rhs in zip(eqs_coeffs, eqs_rhs):
        sym = next(iter(eq_coeff), None)
        sub_coeff, sub_rhs = subsystems[sym2comp[sym]]
        sub_coeff.append(eq_coeff)
        sub_rhs.append(eq_rhs)
    sol = {}
    for subsystem in subsystems:
        subsol = _solve_lin_sys_component(subsystem[0], subsystem[1], ring)
        if subsol is None:
            return None
        sol.update(subsol)
    return sol
from sympy.core import S, pi, Rational
from sympy.functions import assoc_laguerre, sqrt, exp, factorial, factorial2
def E_nl(n, l, hw):
    """
    Returns the Energy of an isotropic harmonic oscillator.

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    l :
        The orbital angular momentum.
    hw :
        The harmonic oscillator parameter.

    Notes
    =====

    The unit of the returned value matches the unit of hw, since the energy is
    calculated as:

        E_nl = (2*n + l + 3/2)*hw

    Examples
    ========

    >>> from sympy.physics.sho import E_nl
    >>> from sympy import symbols
    >>> x, y, z = symbols('x, y, z')
    >>> E_nl(x, y, z)
    z*(2*x + y + 3/2)
    """
    return (2 * n + l + Rational(3, 2)) * hw
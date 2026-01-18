from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.constants import hbar

    Returns the energy of the state corresponding to quantum number ``n``.

    E=(n**2 * (hcross)**2) / (2 * m * r**2)

    Parameters
    ==========

    n :
        The quantum number.
    m :
        Mass of the particle.
    r :
        Radius of circle.

    Examples
    ========

    >>> from sympy.physics.pring import energy
    >>> from sympy import Symbol
    >>> m=Symbol("m")
    >>> r=Symbol("r")
    >>> energy(1, m, r)
    hbar**2/(2*m*r**2)
    >>> energy(2, m, r)
    2*hbar**2/(m*r**2)
    >>> energy(-2, 2.0, 3.0)
    0.111111111111111*hbar**2

    References
    ==========

    .. [1] Atkins, Peter W.; Friedman, Ronald (2005). Molecular Quantum
           Mechanics (4th ed.).  Pages 71-73.

    
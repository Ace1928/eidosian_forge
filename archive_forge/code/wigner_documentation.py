from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableMatrix
from sympy.utilities.misc import as_int
Return the Wigner D matrix for angular momentum J.

    Explanation
    ===========

    J :
        An integer, half-integer, or SymPy symbol for the total angular
        momentum of the angular momentum space being rotated.
    alpha, beta, gamma - Real numbers representing the Euler.
        Angles of rotation about the so-called vertical, line of nodes, and
        figure axes. See [Edmonds74]_.

    Returns
    =======

    A matrix representing the corresponding Euler angle rotation( in the basis
    of eigenvectors of `J_z`).

    .. math ::
        \mathcal{D}_{\alpha \beta \gamma} =
        \exp\big( \frac{i\alpha}{\hbar} J_z\big)
        \exp\big( \frac{i\beta}{\hbar} J_y\big)
        \exp\big( \frac{i\gamma}{\hbar} J_z\big)

    The components are calculated using the general form [Edmonds74]_,
    equation 4.1.12.

    Examples
    ========

    The simplest possible example:

    >>> from sympy.physics.wigner import wigner_d
    >>> from sympy import Integer, symbols, pprint
    >>> half = 1/Integer(2)
    >>> alpha, beta, gamma = symbols("alpha, beta, gamma", real=True)
    >>> pprint(wigner_d(half, alpha, beta, gamma), use_unicode=True)
    ⎡  ⅈ⋅α  ⅈ⋅γ             ⅈ⋅α  -ⅈ⋅γ         ⎤
    ⎢  ───  ───             ───  ─────        ⎥
    ⎢   2    2     ⎛β⎞       2     2      ⎛β⎞ ⎥
    ⎢ ℯ   ⋅ℯ   ⋅cos⎜─⎟     ℯ   ⋅ℯ     ⋅sin⎜─⎟ ⎥
    ⎢              ⎝2⎠                    ⎝2⎠ ⎥
    ⎢                                         ⎥
    ⎢  -ⅈ⋅α   ⅈ⋅γ          -ⅈ⋅α   -ⅈ⋅γ        ⎥
    ⎢  ─────  ───          ─────  ─────       ⎥
    ⎢    2     2     ⎛β⎞     2      2      ⎛β⎞⎥
    ⎢-ℯ     ⋅ℯ   ⋅sin⎜─⎟  ℯ     ⋅ℯ     ⋅cos⎜─⎟⎥
    ⎣                ⎝2⎠                   ⎝2⎠⎦

    
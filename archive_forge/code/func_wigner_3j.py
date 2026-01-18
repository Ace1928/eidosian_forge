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
def wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3):
    """
    Calculate the Wigner 3j symbol `\\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)`.

    Parameters
    ==========

    j_1, j_2, j_3, m_1, m_2, m_3 :
        Integer or half integer.

    Returns
    =======

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_3j
    >>> wigner_3j(2, 6, 4, 0, 0, 0)
    sqrt(715)/143
    >>> wigner_3j(2, 6, 4, 0, 0, 1)
    0

    It is an error to have arguments that are not integer or half
    integer values::

        sage: wigner_3j(2.1, 6, 4, 0, 0, 0)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer
        sage: wigner_3j(2, 6, 4, 1, 0, -1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer or half integer

    Notes
    =====

    The Wigner 3j symbol obeys the following symmetry rules:

    - invariant under any permutation of the columns (with the
      exception of a sign change where `J:=j_1+j_2+j_3`):

      .. math::

         \\begin{aligned}
         \\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
          &=\\operatorname{Wigner3j}(j_3,j_1,j_2,m_3,m_1,m_2) \\\\
          &=\\operatorname{Wigner3j}(j_2,j_3,j_1,m_2,m_3,m_1) \\\\
          &=(-1)^J \\operatorname{Wigner3j}(j_3,j_2,j_1,m_3,m_2,m_1) \\\\
          &=(-1)^J \\operatorname{Wigner3j}(j_1,j_3,j_2,m_1,m_3,m_2) \\\\
          &=(-1)^J \\operatorname{Wigner3j}(j_2,j_1,j_3,m_2,m_1,m_3)
         \\end{aligned}

    - invariant under space inflection, i.e.

      .. math::

         \\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
         =(-1)^J \\operatorname{Wigner3j}(j_1,j_2,j_3,-m_1,-m_2,-m_3)

    - symmetric with respect to the 72 additional symmetries based on
      the work by [Regge58]_

    - zero for `j_1`, `j_2`, `j_3` not fulfilling triangle relation

    - zero for `m_1 + m_2 + m_3 \\neq 0`

    - zero for violating any one of the conditions
      `j_1 \\ge |m_1|`,  `j_2 \\ge |m_2|`,  `j_3 \\ge |m_3|`

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    Authors
    =======

    - Jens Rasch (2009-03-24): initial version
    """
    if int(j_1 * 2) != j_1 * 2 or int(j_2 * 2) != j_2 * 2 or int(j_3 * 2) != j_3 * 2:
        raise ValueError('j values must be integer or half integer')
    if int(m_1 * 2) != m_1 * 2 or int(m_2 * 2) != m_2 * 2 or int(m_3 * 2) != m_3 * 2:
        raise ValueError('m values must be integer or half integer')
    if m_1 + m_2 + m_3 != 0:
        return S.Zero
    prefid = Integer((-1) ** int(j_1 - j_2 - m_3))
    m_3 = -m_3
    a1 = j_1 + j_2 - j_3
    if a1 < 0:
        return S.Zero
    a2 = j_1 - j_2 + j_3
    if a2 < 0:
        return S.Zero
    a3 = -j_1 + j_2 + j_3
    if a3 < 0:
        return S.Zero
    if abs(m_1) > j_1 or abs(m_2) > j_2 or abs(m_3) > j_3:
        return S.Zero
    maxfact = max(j_1 + j_2 + j_3 + 1, j_1 + abs(m_1), j_2 + abs(m_2), j_3 + abs(m_3))
    _calc_factlist(int(maxfact))
    argsqrt = Integer(_Factlist[int(j_1 + j_2 - j_3)] * _Factlist[int(j_1 - j_2 + j_3)] * _Factlist[int(-j_1 + j_2 + j_3)] * _Factlist[int(j_1 - m_1)] * _Factlist[int(j_1 + m_1)] * _Factlist[int(j_2 - m_2)] * _Factlist[int(j_2 + m_2)] * _Factlist[int(j_3 - m_3)] * _Factlist[int(j_3 + m_3)]) / _Factlist[int(j_1 + j_2 + j_3 + 1)]
    ressqrt = sqrt(argsqrt)
    if ressqrt.is_complex or ressqrt.is_infinite:
        ressqrt = ressqrt.as_real_imag()[0]
    imin = max(-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0)
    imax = min(j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3)
    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * _Factlist[int(ii + j_3 - j_1 - m_2)] * _Factlist[int(j_2 + m_2 - ii)] * _Factlist[int(j_1 - ii - m_1)] * _Factlist[int(ii + j_3 - j_2 + m_1)] * _Factlist[int(j_1 + j_2 - j_3 - ii)]
        sumres = sumres + Integer((-1) ** ii) / den
    res = ressqrt * sumres * prefid
    return res
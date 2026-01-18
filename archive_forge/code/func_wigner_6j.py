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
def wigner_6j(j_1, j_2, j_3, j_4, j_5, j_6, prec=None):
    """
    Calculate the Wigner 6j symbol `\\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)`.

    Parameters
    ==========

    j_1, ..., j_6 :
        Integer or half integer.
    prec :
        Precision, default: ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_6j
    >>> wigner_6j(3,3,3,3,3,3)
    -1/14
    >>> wigner_6j(5,5,5,5,5,5)
    1/52

    It is an error to have arguments that are not integer or half
    integer values or do not fulfill the triangle relation::

        sage: wigner_6j(2.5,2.5,2.5,2.5,2.5,2.5)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation
        sage: wigner_6j(0.5,0.5,1.1,0.5,0.5,1.1)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation

    Notes
    =====

    The Wigner 6j symbol is related to the Racah symbol but exhibits
    more symmetries as detailed below.

    .. math::

       \\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
        =(-1)^{j_1+j_2+j_4+j_5} W(j_1,j_2,j_5,j_4,j_3,j_6)

    The Wigner 6j symbol obeys the following symmetry rules:

    - Wigner 6j symbols are left invariant under any permutation of
      the columns:

      .. math::

         \\begin{aligned}
         \\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
          &=\\operatorname{Wigner6j}(j_3,j_1,j_2,j_6,j_4,j_5) \\\\
          &=\\operatorname{Wigner6j}(j_2,j_3,j_1,j_5,j_6,j_4) \\\\
          &=\\operatorname{Wigner6j}(j_3,j_2,j_1,j_6,j_5,j_4) \\\\
          &=\\operatorname{Wigner6j}(j_1,j_3,j_2,j_4,j_6,j_5) \\\\
          &=\\operatorname{Wigner6j}(j_2,j_1,j_3,j_5,j_4,j_6)
         \\end{aligned}

    - They are invariant under the exchange of the upper and lower
      arguments in each of any two columns, i.e.

      .. math::

         \\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
          =\\operatorname{Wigner6j}(j_1,j_5,j_6,j_4,j_2,j_3)
          =\\operatorname{Wigner6j}(j_4,j_2,j_6,j_1,j_5,j_3)
          =\\operatorname{Wigner6j}(j_4,j_5,j_3,j_1,j_2,j_6)

    - additional 6 symmetries [Regge59]_ giving rise to 144 symmetries
      in total

    - only non-zero if any triple of `j`'s fulfill a triangle relation

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 6j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    """
    res = (-1) ** int(j_1 + j_2 + j_4 + j_5) * racah(j_1, j_2, j_5, j_4, j_3, j_6, prec)
    return res
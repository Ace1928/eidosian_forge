from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.relational import is_eq
from sympy.functions.elementary.complexes import (conjugate, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log as ln)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.trigsimp import trigsimp
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.core.sympify import sympify, _sympify
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_not, fuzzy_or
from mpmath.libmp.libmpf import prec_to_dps
@staticmethod
def _generic_mul(q1, q2):
    """Generic multiplication.

        Parameters
        ==========

        q1 : Quaternion or symbol
        q2 : Quaternion or symbol

        It is important to note that if neither q1 nor q2 is a Quaternion,
        this function simply returns q1 * q2.

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying q1 and q2

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import Symbol, S
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> Quaternion._generic_mul(q1, q2)
        (-60) + 12*i + 30*j + 24*k
        >>> Quaternion._generic_mul(q1, S(2))
        2 + 4*i + 6*j + 8*k
        >>> x = Symbol('x', real = True)
        >>> Quaternion._generic_mul(q1, x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields :

        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> Quaternion._generic_mul(q3, 2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
    if not isinstance(q1, Quaternion) and (not isinstance(q2, Quaternion)):
        return q1 * q2
    if not isinstance(q1, Quaternion):
        if q2.real_field and q1.is_complex:
            return Quaternion(re(q1), im(q1), 0, 0) * q2
        elif q1.is_commutative:
            return Quaternion(q1 * q2.a, q1 * q2.b, q1 * q2.c, q1 * q2.d)
        else:
            raise ValueError('Only commutative expressions can be multiplied with a Quaternion.')
    if not isinstance(q2, Quaternion):
        if q1.real_field and q2.is_complex:
            return q1 * Quaternion(re(q2), im(q2), 0, 0)
        elif q2.is_commutative:
            return Quaternion(q2 * q1.a, q2 * q1.b, q2 * q1.c, q2 * q1.d)
        else:
            raise ValueError('Only commutative expressions can be multiplied with a Quaternion.')
    if q1._norm is None and q2._norm is None:
        norm = None
    else:
        norm = q1.norm() * q2.norm()
    return Quaternion(-q1.b * q2.b - q1.c * q2.c - q1.d * q2.d + q1.a * q2.a, q1.b * q2.a + q1.c * q2.d - q1.d * q2.c + q1.a * q2.b, -q1.b * q2.d + q1.c * q2.a + q1.d * q2.b + q1.a * q2.c, q1.b * q2.c - q1.c * q2.b + q1.d * q2.a + q1.a * q2.d, norm=norm)
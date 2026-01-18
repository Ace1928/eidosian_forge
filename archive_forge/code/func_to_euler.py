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
def to_euler(self, seq, angle_addition=True, avoid_square_root=False):
    """Returns Euler angles representing same rotation as the quaternion,
        in the sequence given by ``seq``. This implements the method described
        in [1]_.

        For degenerate cases (gymbal lock cases), the third angle is
        set to zero.

        Parameters
        ==========

        seq : string of length 3
            Represents the sequence of rotations.
            For intrinsic rotations, seq must be all lowercase and its elements
            must be from the set ``{'x', 'y', 'z'}``
            For extrinsic rotations, seq must be all uppercase and its elements
            must be from the set ``{'X', 'Y', 'Z'}``

        angle_addition : bool
            When True, first and third angles are given as an addition and
            subtraction of two simpler ``atan2`` expressions. When False, the
            first and third angles are each given by a single more complicated
            ``atan2`` expression. This equivalent expression is given by:

            .. math::

                \\operatorname{atan_2} (b,a) \\pm \\operatorname{atan_2} (d,c) =
                \\operatorname{atan_2} (bc\\pm ad, ac\\mp bd)

            Default value: True

        avoid_square_root : bool
            When True, the second angle is calculated with an expression based
            on ``acos``, which is slightly more complicated but avoids a square
            root. When False, second angle is calculated with ``atan2``, which
            is simpler and can be better for numerical reasons (some
            numerical implementations of ``acos`` have problems near zero).
            Default value: False


        Returns
        =======

        Tuple
            The Euler angles calculated from the quaternion

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> euler = Quaternion(a, b, c, d).to_euler('zyz')
        >>> euler
        (-atan2(-b, c) + atan2(d, a),
         2*atan2(sqrt(b**2 + c**2), sqrt(a**2 + d**2)),
         atan2(-b, c) + atan2(d, a))


        References
        ==========

        .. [1] https://doi.org/10.1371/journal.pone.0276302

        """
    if self.is_zero_quaternion():
        raise ValueError('Cannot convert a quaternion with norm 0.')
    angles = [0, 0, 0]
    extrinsic = _is_extrinsic(seq)
    i, j, k = seq.lower()
    i = 'xyz'.index(i) + 1
    j = 'xyz'.index(j) + 1
    k = 'xyz'.index(k) + 1
    if not extrinsic:
        i, k = (k, i)
    symmetric = i == k
    if symmetric:
        k = 6 - i - j
    sign = (i - j) * (j - k) * (k - i) // 2
    elements = [self.a, self.b, self.c, self.d]
    a = elements[0]
    b = elements[i]
    c = elements[j]
    d = elements[k] * sign
    if not symmetric:
        a, b, c, d = (a - c, b + d, c + a, d - b)
    if avoid_square_root:
        if symmetric:
            n2 = self.norm() ** 2
            angles[1] = acos((a * a + b * b - c * c - d * d) / n2)
        else:
            n2 = 2 * self.norm() ** 2
            angles[1] = asin((c * c + d * d - a * a - b * b) / n2)
    else:
        angles[1] = 2 * atan2(sqrt(c * c + d * d), sqrt(a * a + b * b))
        if not symmetric:
            angles[1] -= S.Pi / 2
    case = 0
    if is_eq(c, S.Zero) and is_eq(d, S.Zero):
        case = 1
    if is_eq(a, S.Zero) and is_eq(b, S.Zero):
        case = 2
    if case == 0:
        if angle_addition:
            angles[0] = atan2(b, a) + atan2(d, c)
            angles[2] = atan2(b, a) - atan2(d, c)
        else:
            angles[0] = atan2(b * c + a * d, a * c - b * d)
            angles[2] = atan2(b * c - a * d, a * c + b * d)
    else:
        angles[2 * (not extrinsic)] = S.Zero
        if case == 1:
            angles[2 * extrinsic] = 2 * atan2(b, a)
        else:
            angles[2 * extrinsic] = 2 * atan2(d, c)
            angles[2 * extrinsic] *= -1 if extrinsic else 1
    if not symmetric:
        angles[0] *= sign
    if extrinsic:
        return tuple(angles[::-1])
    else:
        return tuple(angles)
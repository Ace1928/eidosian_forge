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
@classmethod
def vector_coplanar(cls, q1, q2, q3):
    """
        Returns True if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are coplanar.

        Explanation
        ===========

        Three pure quaternions are vector coplanar if the quaternions seen as 3D vectors are coplanar.

        Parameters
        ==========

        q1
            A pure Quaternion.
        q2
            A pure Quaternion.
        q3
            A pure Quaternion.

        Returns
        =======

        True : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are coplanar.
        False : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are not coplanar.
        None : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are coplanar is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q1 = Quaternion(0, 4, 4, 4)
        >>> q2 = Quaternion(0, 8, 8, 8)
        >>> q3 = Quaternion(0, 24, 24, 24)
        >>> Quaternion.vector_coplanar(q1, q2, q3)
        True

        >>> q1 = Quaternion(0, 8, 16, 8)
        >>> q2 = Quaternion(0, 8, 3, 12)
        >>> Quaternion.vector_coplanar(q1, q2, q3)
        False

        See Also
        ========

        axis
        is_pure

        """
    if fuzzy_not(q1.is_pure()) or fuzzy_not(q2.is_pure()) or fuzzy_not(q3.is_pure()):
        raise ValueError('The given quaternions must be pure')
    M = Matrix([[q1.b, q1.c, q1.d], [q2.b, q2.c, q2.d], [q3.b, q3.c, q3.d]]).det()
    return M.is_zero
from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core import S, Dummy, Lambda
from sympy.core.symbol import Str
from sympy.core.symbol import symbols
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.matrices.matrices import MatrixBase
from sympy.solvers import solve
from sympy.vector.scalar import BaseScalar
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from sympy.matrices.dense import eye
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
import sympy.vector
from sympy.vector.orienters import (Orienter, AxisOrienter, BodyOrienter,
from sympy.vector.vector import BaseVector
@staticmethod
def _check_orthogonality(equations):
    """
        Helper method for _connect_to_cartesian. It checks if
        set of transformation equations create orthogonal curvilinear
        coordinate system

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations

        """
    x1, x2, x3 = symbols('x1, x2, x3', cls=Dummy)
    equations = equations(x1, x2, x3)
    v1 = Matrix([diff(equations[0], x1), diff(equations[1], x1), diff(equations[2], x1)])
    v2 = Matrix([diff(equations[0], x2), diff(equations[1], x2), diff(equations[2], x2)])
    v3 = Matrix([diff(equations[0], x3), diff(equations[1], x3), diff(equations[2], x3)])
    if any((simplify(i[0] + i[1] + i[2]) == 0 for i in (v1, v2, v3))):
        return False
    elif simplify(v1.dot(v2)) == 0 and simplify(v2.dot(v3)) == 0 and (simplify(v3.dot(v1)) == 0):
        return True
    else:
        return False
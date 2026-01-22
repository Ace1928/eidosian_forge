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
def _set_inv_trans_equations(curv_coord_name):
    """
        Store information about inverse transformation equations for
        pre-defined coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
    if curv_coord_name == 'cartesian':
        return lambda x, y, z: (x, y, z)
    if curv_coord_name == 'spherical':
        return lambda x, y, z: (sqrt(x ** 2 + y ** 2 + z ** 2), acos(z / sqrt(x ** 2 + y ** 2 + z ** 2)), atan2(y, x))
    if curv_coord_name == 'cylindrical':
        return lambda x, y, z: (sqrt(x ** 2 + y ** 2), atan2(y, x), z)
    raise ValueError('Wrong set of parameters.Type of coordinate system is defined')
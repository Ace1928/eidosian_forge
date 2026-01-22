from sympy.core.basic import Basic
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, rot_axis1, rot_axis2, rot_axis3)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.core.cache import cacheit
from sympy.core.symbol import Str
import sympy.vector
class AxisOrienter(Orienter):
    """
    Class to denote an axis orienter.
    """

    def __new__(cls, angle, axis):
        if not isinstance(axis, sympy.vector.Vector):
            raise TypeError('axis should be a Vector')
        angle = sympify(angle)
        obj = super().__new__(cls, angle, axis)
        obj._angle = angle
        obj._axis = axis
        return obj

    def __init__(self, angle, axis):
        """
        Axis rotation is a rotation about an arbitrary axis by
        some angle. The angle is supplied as a SymPy expr scalar, and
        the axis is supplied as a Vector.

        Parameters
        ==========

        angle : Expr
            The angle by which the new system is to be rotated

        axis : Vector
            The axis around which the rotation has to be performed

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> from sympy.vector import AxisOrienter
        >>> orienter = AxisOrienter(q1, N.i + 2 * N.j)
        >>> B = N.orient_new('B', (orienter, ))

        """
        pass

    @cacheit
    def rotation_matrix(self, system):
        """
        The rotation matrix corresponding to this orienter
        instance.

        Parameters
        ==========

        system : CoordSys3D
            The coordinate system wrt which the rotation matrix
            is to be computed
        """
        axis = sympy.vector.express(self.axis, system).normalize()
        axis = axis.to_matrix(system)
        theta = self.angle
        parent_orient = (eye(3) - axis * axis.T) * cos(theta) + Matrix([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) * sin(theta) + axis * axis.T
        parent_orient = parent_orient.T
        return parent_orient

    @property
    def angle(self):
        return self._angle

    @property
    def axis(self):
        return self._axis
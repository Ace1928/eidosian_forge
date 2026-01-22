from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def ang_vel_in(self, otherframe):
    """Returns the angular velocity Vector of the ReferenceFrame.

        Effectively returns the Vector:

        ^N omega ^B

        which represent the angular velocity of B in N, where B is self, and
        N is otherframe.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The ReferenceFrame which the angular velocity is returned in.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> V = 10 * N.x
        >>> A.set_ang_vel(N, V)
        >>> A.ang_vel_in(N)
        10*N.x

        """
    _check_frame(otherframe)
    flist = self._dict_list(otherframe, 1)
    outvec = Vector(0)
    for i in range(len(flist) - 1):
        outvec += flist[i]._ang_vel_dict[flist[i + 1]]
    return outvec
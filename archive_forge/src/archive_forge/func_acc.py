from .vector import Vector, _check_vector
from .frame import _check_frame
from warnings import warn
def acc(self, frame):
    """The acceleration Vector of this Point in a ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which the returned acceleration vector will be defined
            in.

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_acc(N, 10 * N.x)
        >>> p1.acc(N)
        10*N.x

        """
    _check_frame(frame)
    if not frame in self._acc_dict:
        if self.vel(frame) != 0:
            return self._vel_dict[frame].dt(frame)
        else:
            return Vector(0)
    return self._acc_dict[frame]
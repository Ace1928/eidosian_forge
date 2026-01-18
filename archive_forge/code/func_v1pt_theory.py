from .vector import Vector, _check_vector
from .frame import _check_frame
from warnings import warn
def v1pt_theory(self, otherpoint, outframe, interframe):
    """Sets the velocity of this point with the 1-point theory.

        The 1-point theory for point velocity looks like this:

        ^N v^P = ^B v^P + ^N v^O + ^N omega^B x r^OP

        where O is a point fixed in B, P is a point moving in B, and B is
        rotating in frame N.

        Parameters
        ==========

        otherpoint : Point
            The first point of the 1-point theory (O)
        outframe : ReferenceFrame
            The frame we want this point's velocity defined in (N)
        interframe : ReferenceFrame
            The intermediate frame in this calculation (B)

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> from sympy.physics.vector import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q = dynamicsymbols('q')
        >>> q2 = dynamicsymbols('q2')
        >>> qd = dynamicsymbols('q', 1)
        >>> q2d = dynamicsymbols('q2', 1)
        >>> N = ReferenceFrame('N')
        >>> B = ReferenceFrame('B')
        >>> B.set_ang_vel(N, 5 * B.y)
        >>> O = Point('O')
        >>> P = O.locatenew('P', q * B.x)
        >>> P.set_vel(B, qd * B.x + q2d * B.y)
        >>> O.set_vel(N, 0)
        >>> P.v1pt_theory(O, N, B)
        q'*B.x + q2'*B.y - 5*q*B.z

        """
    _check_frame(outframe)
    _check_frame(interframe)
    self._check_point(otherpoint)
    dist = self.pos_from(otherpoint)
    v1 = self.vel(interframe)
    v2 = otherpoint.vel(outframe)
    omega = interframe.ang_vel_in(outframe)
    self.set_vel(outframe, v1 + v2 + (omega ^ dist))
    return self.vel(outframe)
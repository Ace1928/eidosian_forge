from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
def _curve_to(self, k):
    """
        Compute the two control points for a nice cubic curve from the
        kth spline knot to the next one.  Return the kth spline knot
        and the two control points.  We do not allow the speed at the
        spline knots to exceed the distance to the interlacing vertex
        of the PL curve; this avoids extraneous inflection points.
        """
    A, B = self.spline_knots[k:k + 2]
    vA, vB = self.tangents[k:k + 2]
    A_speed_max, B_speed_max = (abs(vA), abs(vB))
    base = B - A
    l, psi = (abs(base), base.angle())
    theta, phi = (vA.angle() - psi, psi - vB.angle())
    ctheta, stheta = (cos(theta), sin(theta))
    cphi, sphi = (cos(phi), sin(phi))
    a = sqrt(2.0)
    b = 1.0 / 16.0
    c = (3.0 - sqrt(5.0)) / 2.0
    alpha = a * (stheta - b * sphi) * (sphi - b * stheta) * (ctheta - cphi)
    rho = (2 + alpha) / ((1 + (1 - c) * ctheta + c * cphi) * self.tension1)
    sigma = (2 - alpha) / ((1 + (1 - c) * cphi + c * ctheta) * self.tension2)
    A_speed = min(l * rho / 3, A_speed_max)
    B_speed = min(l * sigma / 3, B_speed_max)
    return [A, A + self._polar_to_vector(A_speed, psi + theta), B - self._polar_to_vector(B_speed, psi - phi)]
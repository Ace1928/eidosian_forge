from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import raises
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.integrals import ParametricIntegral, vector_integrate
from sympy.vector.parametricregion import ParametricRegion
from sympy.vector.implicitregion import ImplicitRegion
from sympy.abc import x, y, z, u, v, r, t, theta, phi
from sympy.geometry import Point, Segment, Curve, Circle, Polygon, Plane
def test_parametric_volumeintegrals():
    cube = ParametricRegion((x, y, z), (x, 0, 1), (y, 0, 1), (z, 0, 1))
    assert ParametricIntegral(1, cube) == 1
    solidsphere1 = ParametricRegion((r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi)), (r, 0, 2), (theta, 0, 2 * pi), (phi, 0, pi))
    solidsphere2 = ParametricRegion((r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi)), (r, 0, 2), (phi, 0, pi), (theta, 0, 2 * pi))
    assert ParametricIntegral(C.x ** 2 + C.y ** 2, solidsphere1) == -256 * pi / 15
    assert ParametricIntegral(C.x ** 2 + C.y ** 2, solidsphere2) == 256 * pi / 15
    region_under_plane1 = ParametricRegion((x, y, z), (x, 0, 3), (y, 0, -2 * x / 3 + 2), (z, 0, 6 - 2 * x - 3 * y))
    region_under_plane2 = ParametricRegion((x, y, z), (x, 0, 3), (z, 0, 6 - 2 * x - 3 * y), (y, 0, -2 * x / 3 + 2))
    assert ParametricIntegral(C.x * C.i + C.j - 100 * C.k, region_under_plane1) == ParametricIntegral(C.x * C.i + C.j - 100 * C.k, region_under_plane2)
    assert ParametricIntegral(2 * C.x, region_under_plane2) == -9
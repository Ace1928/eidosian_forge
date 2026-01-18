from sympy.core.numbers import comp, Rational
from sympy.physics.optics.utils import (refraction_angle, fresnel_coefficients,
from sympy.physics.optics.medium import Medium
from sympy.physics.units import e0
from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.geometry.point import Point3D
from sympy.geometry.line import Ray3D
from sympy.geometry.plane import Plane
from sympy.testing.pytest import raises
def test_transverse_magnification():
    si, so = symbols('si, so')
    assert transverse_magnification(si, so) == -si / so
    assert transverse_magnification(30, 15) == -2
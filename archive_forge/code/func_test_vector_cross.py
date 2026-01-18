from sympy.core import Rational, S
from sympy.simplify import simplify, trigsimp
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.vector.vector import Vector, BaseVector, VectorAdd, \
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.vector import Cross, Dot, cross
from sympy.testing.pytest import raises
def test_vector_cross():
    assert i.cross(Vector.zero) == Vector.zero
    assert Vector.zero.cross(i) == Vector.zero
    assert i.cross(i) == Vector.zero
    assert i.cross(j) == k
    assert i.cross(k) == -j
    assert i ^ i == Vector.zero
    assert i ^ j == k
    assert i ^ k == -j
    assert j.cross(i) == -k
    assert j.cross(j) == Vector.zero
    assert j.cross(k) == i
    assert j ^ i == -k
    assert j ^ j == Vector.zero
    assert j ^ k == i
    assert k.cross(i) == j
    assert k.cross(j) == -i
    assert k.cross(k) == Vector.zero
    assert k ^ i == j
    assert k ^ j == -i
    assert k ^ k == Vector.zero
    assert k.cross(1) == Cross(k, 1)
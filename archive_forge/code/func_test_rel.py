from symengine.test_utilities import raises
from symengine import (Symbol, sin, cos, Integer, Add, I, RealDouble, ComplexDouble, sqrt)
from unittest.case import SkipTest
def test_rel():
    x = Symbol('x')
    y = Symbol('y')
    ex = x + y < x
    assert repr(ex) == 'x + y < x'
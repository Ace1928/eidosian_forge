import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_shear_constructor(self):
    shear = Affine.shear(30)
    assert isinstance(shear, Affine)
    mx = math.tan(math.radians(30))
    seq_almost_equal(tuple(shear), (1, mx, 0, 0, 1, 0, 0, 0, 1))
    shear = Affine.shear(-15, 60)
    mx = math.tan(math.radians(-15))
    my = math.tan(math.radians(60))
    seq_almost_equal(tuple(shear), (1, mx, 0, my, 1, 0, 0, 0, 1))
    shear = Affine.shear(y_angle=45)
    seq_almost_equal(tuple(shear), (1, 0, 0, 1, 1, 0, 0, 0, 1))
import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_rotation_angle():
    assert Affine.identity().rotation_angle == 0.0
    assert Affine.scale(2).rotation_angle == 0.0
    assert Affine.scale(2, 1).rotation_angle == 0.0
    assert Affine.translation(32, -47).rotation_angle == pytest.approx(0.0)
    assert Affine.rotation(30).rotation_angle == pytest.approx(30)
    assert Affine.rotation(-150).rotation_angle == pytest.approx(-150)
import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_rotation_constructor_quadrants(self):
    assert tuple(Affine.rotation(0)) == (1, 0, 0, 0, 1, 0, 0, 0, 1)
    assert tuple(Affine.rotation(90)) == (0, -1, 0, 1, 0, 0, 0, 0, 1)
    assert tuple(Affine.rotation(180)) == (-1, 0, 0, 0, -1, 0, 0, 0, 1)
    assert tuple(Affine.rotation(-180)) == (-1, 0, 0, 0, -1, 0, 0, 0, 1)
    assert tuple(Affine.rotation(270)) == (0, 1, 0, -1, 0, 0, 0, 0, 1)
    assert tuple(Affine.rotation(-90)) == (0, 1, 0, -1, 0, 0, 0, 0, 1)
    assert tuple(Affine.rotation(360)) == (1, 0, 0, 0, 1, 0, 0, 0, 1)
    assert tuple(Affine.rotation(450)) == (0, -1, 0, 1, 0, 0, 0, 0, 1)
    assert tuple(Affine.rotation(-450)) == (0, 1, 0, -1, 0, 0, 0, 0, 1)
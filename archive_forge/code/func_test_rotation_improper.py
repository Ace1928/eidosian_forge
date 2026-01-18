import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_rotation_improper():
    with pytest.raises(affine.UndefinedRotationError):
        Affine.scale(-1, 1).rotation_angle
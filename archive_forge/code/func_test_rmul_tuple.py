import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_rmul_tuple():
    with pytest.warns(DeprecationWarning):
        t = Affine(1, 2, 3, 4, 5, 6)
        (2.0, 2.0) * t
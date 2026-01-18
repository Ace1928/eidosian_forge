import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_simple_world(self):
    s = '1.0\n0.0\n0.0\n-1.0\n100.5\n199.5\n'
    a = affine.loadsw(s)
    assert a == Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
    assert affine.dumpsw(a) == s
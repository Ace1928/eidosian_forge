import pytest
from mpmath import *
def test_quadgl_linear():
    assert quadgl(lambda x: x, [0, 1], maxdegree=1).ae(0.5)
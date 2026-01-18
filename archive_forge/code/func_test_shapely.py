import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_shapely():
    t = Affine(425.0, 0.0, -237481.5, 0.0, -425.0, 237536.4)
    assert t.to_shapely() == (425.0, 0.0, 0.0, -425, -237481.5, 237536.4)
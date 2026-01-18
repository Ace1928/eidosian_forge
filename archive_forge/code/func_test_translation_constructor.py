import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_translation_constructor(self):
    trans = Affine.translation(2, -5)
    assert isinstance(trans, Affine)
    assert tuple(trans) == (1, 0, 2, 0, 1, -5, 0, 0, 1)
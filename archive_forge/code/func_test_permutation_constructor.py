import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_permutation_constructor(self):
    perm = Affine.permutation()
    assert isinstance(perm, Affine)
    assert tuple(perm) == (0, 1, 0, 1, 0, 0, 0, 0, 1)
    assert (perm * perm).is_identity
import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_rotation_contructor_wrong_arg_types(self):
    with pytest.raises(TypeError):
        Affine.rotation(1, 1)
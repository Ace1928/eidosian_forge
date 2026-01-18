import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_wrong_arg_type(self):
    with pytest.raises(TypeError):
        Affine(None)
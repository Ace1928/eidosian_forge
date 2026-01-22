import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
class TestStringAssumedLength(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'string', 'gh24008.f')]

    def test_gh24008(self):
        self.module.greet('joe', 'bob')
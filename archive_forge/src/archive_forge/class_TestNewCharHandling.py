import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
@pytest.mark.slow
class TestNewCharHandling(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'string', 'gh25286.pyf'), util.getpath('tests', 'src', 'string', 'gh25286.f90')]
    module_name = '_char_handling_test'

    def test_gh25286(self):
        info = self.module.charint('T')
        assert info == 2
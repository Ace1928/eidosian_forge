import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class Test_sctype2char:

    def test_scalar_type(self):
        assert_equal(np.sctype2char(np.double), 'd')
        assert_equal(np.sctype2char(np.int_), 'l')
        assert_equal(np.sctype2char(np.str_), 'U')
        assert_equal(np.sctype2char(np.bytes_), 'S')

    def test_other_type(self):
        assert_equal(np.sctype2char(float), 'd')
        assert_equal(np.sctype2char(list), 'O')
        assert_equal(np.sctype2char(np.ndarray), 'O')

    def test_third_party_scalar_type(self):
        from numpy.core._rational_tests import rational
        assert_raises(KeyError, np.sctype2char, rational)
        assert_raises(KeyError, np.sctype2char, rational(1))

    def test_array_instance(self):
        assert_equal(np.sctype2char(np.array([1.0, 2.0])), 'd')

    def test_abstract_type(self):
        assert_raises(KeyError, np.sctype2char, np.floating)

    def test_non_type(self):
        assert_raises(ValueError, np.sctype2char, 1)
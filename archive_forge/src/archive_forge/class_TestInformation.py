import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
class TestInformation:

    def setup_method(self):
        self.A = np.array([[' abc ', ''], ['12345', 'MixedCase'], ['123 \t 345 \x00 ', 'UPPER']]).view(np.chararray)
        self.B = np.array([[' Σ ', ''], ['12345', 'MixedCase'], ['123 \t 345 \x00 ', 'UPPER']]).view(np.chararray)

    def test_len(self):
        assert_(issubclass(np.char.str_len(self.A).dtype.type, np.integer))
        assert_array_equal(np.char.str_len(self.A), [[5, 0], [5, 9], [12, 5]])
        assert_array_equal(np.char.str_len(self.B), [[3, 0], [5, 9], [12, 5]])

    def test_count(self):
        assert_(issubclass(self.A.count('').dtype.type, np.integer))
        assert_array_equal(self.A.count('a'), [[1, 0], [0, 1], [0, 0]])
        assert_array_equal(self.A.count('123'), [[0, 0], [1, 0], [1, 0]])
        assert_array_equal(self.A.count('a', 0, 2), [[1, 0], [0, 0], [0, 0]])
        assert_array_equal(self.B.count('a'), [[0, 0], [0, 1], [0, 0]])
        assert_array_equal(self.B.count('123'), [[0, 0], [1, 0], [1, 0]])

    def test_endswith(self):
        assert_(issubclass(self.A.endswith('').dtype.type, np.bool_))
        assert_array_equal(self.A.endswith(' '), [[1, 0], [0, 0], [1, 0]])
        assert_array_equal(self.A.endswith('3', 0, 3), [[0, 0], [1, 0], [1, 0]])

        def fail():
            self.A.endswith('3', 'fdjk')
        assert_raises(TypeError, fail)

    def test_find(self):
        assert_(issubclass(self.A.find('a').dtype.type, np.integer))
        assert_array_equal(self.A.find('a'), [[1, -1], [-1, 6], [-1, -1]])
        assert_array_equal(self.A.find('3'), [[-1, -1], [2, -1], [2, -1]])
        assert_array_equal(self.A.find('a', 0, 2), [[1, -1], [-1, -1], [-1, -1]])
        assert_array_equal(self.A.find(['1', 'P']), [[-1, -1], [0, -1], [0, 1]])

    def test_index(self):

        def fail():
            self.A.index('a')
        assert_raises(ValueError, fail)
        assert_(np.char.index('abcba', 'b') == 1)
        assert_(issubclass(np.char.index('abcba', 'b').dtype.type, np.integer))

    def test_isalnum(self):
        assert_(issubclass(self.A.isalnum().dtype.type, np.bool_))
        assert_array_equal(self.A.isalnum(), [[False, False], [True, True], [False, True]])

    def test_isalpha(self):
        assert_(issubclass(self.A.isalpha().dtype.type, np.bool_))
        assert_array_equal(self.A.isalpha(), [[False, False], [False, True], [False, True]])

    def test_isdigit(self):
        assert_(issubclass(self.A.isdigit().dtype.type, np.bool_))
        assert_array_equal(self.A.isdigit(), [[False, False], [True, False], [False, False]])

    def test_islower(self):
        assert_(issubclass(self.A.islower().dtype.type, np.bool_))
        assert_array_equal(self.A.islower(), [[True, False], [False, False], [False, False]])

    def test_isspace(self):
        assert_(issubclass(self.A.isspace().dtype.type, np.bool_))
        assert_array_equal(self.A.isspace(), [[False, False], [False, False], [False, False]])

    def test_istitle(self):
        assert_(issubclass(self.A.istitle().dtype.type, np.bool_))
        assert_array_equal(self.A.istitle(), [[False, False], [False, False], [False, False]])

    def test_isupper(self):
        assert_(issubclass(self.A.isupper().dtype.type, np.bool_))
        assert_array_equal(self.A.isupper(), [[False, False], [False, False], [False, True]])

    def test_rfind(self):
        assert_(issubclass(self.A.rfind('a').dtype.type, np.integer))
        assert_array_equal(self.A.rfind('a'), [[1, -1], [-1, 6], [-1, -1]])
        assert_array_equal(self.A.rfind('3'), [[-1, -1], [2, -1], [6, -1]])
        assert_array_equal(self.A.rfind('a', 0, 2), [[1, -1], [-1, -1], [-1, -1]])
        assert_array_equal(self.A.rfind(['1', 'P']), [[-1, -1], [0, -1], [0, 2]])

    def test_rindex(self):

        def fail():
            self.A.rindex('a')
        assert_raises(ValueError, fail)
        assert_(np.char.rindex('abcba', 'b') == 3)
        assert_(issubclass(np.char.rindex('abcba', 'b').dtype.type, np.integer))

    def test_startswith(self):
        assert_(issubclass(self.A.startswith('').dtype.type, np.bool_))
        assert_array_equal(self.A.startswith(' '), [[1, 0], [0, 0], [0, 0]])
        assert_array_equal(self.A.startswith('1', 0, 3), [[0, 0], [1, 0], [1, 0]])

        def fail():
            self.A.startswith('3', 'fdjk')
        assert_raises(TypeError, fail)
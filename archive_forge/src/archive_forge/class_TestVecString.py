import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
class TestVecString:

    def test_non_existent_method(self):

        def fail():
            _vec_string('a', np.bytes_, 'bogus')
        assert_raises(AttributeError, fail)

    def test_non_string_array(self):

        def fail():
            _vec_string(1, np.bytes_, 'strip')
        assert_raises(TypeError, fail)

    def test_invalid_args_tuple(self):

        def fail():
            _vec_string(['a'], np.bytes_, 'strip', 1)
        assert_raises(TypeError, fail)

    def test_invalid_type_descr(self):

        def fail():
            _vec_string(['a'], 'BOGUS', 'strip')
        assert_raises(TypeError, fail)

    def test_invalid_function_args(self):

        def fail():
            _vec_string(['a'], np.bytes_, 'strip', (1,))
        assert_raises(TypeError, fail)

    def test_invalid_result_type(self):

        def fail():
            _vec_string(['a'], np.int_, 'strip')
        assert_raises(TypeError, fail)

    def test_broadcast_error(self):

        def fail():
            _vec_string([['abc', 'def']], np.int_, 'find', (['a', 'd', 'j'],))
        assert_raises(ValueError, fail)
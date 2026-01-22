import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class TestStringEqual:

    def test_simple(self):
        assert_string_equal('hello', 'hello')
        assert_string_equal('hello\nmultiline', 'hello\nmultiline')
        with pytest.raises(AssertionError) as exc_info:
            assert_string_equal('foo\nbar', 'hello\nbar')
        msg = str(exc_info.value)
        assert_equal(msg, 'Differences in strings:\n- foo\n+ hello')
        assert_raises(AssertionError, lambda: assert_string_equal('foo', 'hello'))

    def test_regex(self):
        assert_string_equal('a+*b', 'a+*b')
        assert_raises(AssertionError, lambda: assert_string_equal('aaa', 'a+b'))
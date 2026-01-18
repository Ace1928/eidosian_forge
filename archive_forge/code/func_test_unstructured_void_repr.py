import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_unstructured_void_repr(self):
    a = np.array([27, 91, 50, 75, 7, 65, 10, 8, 27, 91, 51, 49, 109, 82, 101, 100], dtype='u1').view('V8')
    assert_equal(repr(a[0]), "void(b'\\x1B\\x5B\\x32\\x4B\\x07\\x41\\x0A\\x08')")
    assert_equal(str(a[0]), "b'\\x1B\\x5B\\x32\\x4B\\x07\\x41\\x0A\\x08'")
    assert_equal(repr(a), "array([b'\\x1B\\x5B\\x32\\x4B\\x07\\x41\\x0A\\x08',\n       b'\\x1B\\x5B\\x33\\x31\\x6D\\x52\\x65\\x64'], dtype='|V8')")
    assert_equal(eval(repr(a), vars(np)), a)
    assert_equal(eval(repr(a[0]), vars(np)), a[0])
import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
@pytest.mark.xfail(reason='See gh-10544')
def test_object_subclass(self):

    class sub(np.ndarray):

        def __new__(cls, inp):
            obj = np.asarray(inp).view(cls)
            return obj

        def __getitem__(self, ind):
            ret = super().__getitem__(ind)
            return sub(ret)
    x = sub([None, None])
    assert_equal(repr(x), 'sub([None, None], dtype=object)')
    assert_equal(str(x), '[None None]')
    x = sub([None, sub([None, None])])
    assert_equal(repr(x), 'sub([None, sub([None, None], dtype=object)], dtype=object)')
    assert_equal(str(x), '[None sub([None, None], dtype=object)]')
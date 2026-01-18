import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_recarray_stringtypes(self):
    a = np.array([('abc ', 1), ('abc', 2)], dtype=[('foo', 'S4'), ('bar', int)])
    a = a.view(np.recarray)
    assert_equal(a.foo[0] == a.foo[1], False)
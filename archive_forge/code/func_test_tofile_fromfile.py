import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_tofile_fromfile(self):
    with temppath(suffix='.bin') as path:
        path = Path(path)
        np.random.seed(123)
        a = np.random.rand(10).astype('f8,i4,a5')
        a[5] = (0.5, 10, 'abcde')
        with path.open('wb') as fd:
            a.tofile(fd)
        x = np.core.records.fromfile(path, formats='f8,i4,a5', shape=10)
        assert_array_equal(x, a)
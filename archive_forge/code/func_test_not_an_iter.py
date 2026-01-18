import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_not_an_iter(self):
    with pytest.raises(TypeError, match='error reading from object, expected an iterable'):
        np.core._multiarray_umath._load_from_filelike(object(), dtype=np.dtype('i'), filelike=False)
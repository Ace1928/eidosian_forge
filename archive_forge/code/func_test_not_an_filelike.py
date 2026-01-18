import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_not_an_filelike(self):
    with pytest.raises(AttributeError, match='.*read'):
        np.core._multiarray_umath._load_from_filelike(object(), dtype=np.dtype('i'), filelike=True)
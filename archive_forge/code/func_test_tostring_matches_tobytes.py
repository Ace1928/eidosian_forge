import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
def test_tostring_matches_tobytes(self):
    arr = np.array(list(b'test\xff'), dtype=np.uint8)
    b = arr.tobytes()
    with assert_warns(DeprecationWarning):
        s = arr.tostring()
    assert s == b
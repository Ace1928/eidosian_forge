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
def test_npy_pyarrayas2d_deprecation(self):
    from numpy.core._multiarray_tests import npy_pyarrayas2d_deprecation
    assert_raises(NotImplementedError, npy_pyarrayas2d_deprecation)
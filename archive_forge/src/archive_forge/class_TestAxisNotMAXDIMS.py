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
class TestAxisNotMAXDIMS(_DeprecationTestCase):
    message = 'Using `axis=32` \\(MAXDIMS\\) is deprecated'

    def test_deprecated(self):
        a = np.zeros((1,) * 32)
        self.assert_deprecated(lambda: np.repeat(a, 1, axis=np.MAXDIMS))
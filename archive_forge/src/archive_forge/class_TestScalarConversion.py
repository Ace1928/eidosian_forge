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
class TestScalarConversion(_DeprecationTestCase):

    def test_float_conversion(self):
        self.assert_deprecated(float, args=(np.array([3.14]),))

    def test_behaviour(self):
        b = np.array([[3.14]])
        c = np.zeros(5)
        with pytest.warns(DeprecationWarning):
            c[0] = b
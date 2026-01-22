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
class TestNPY_CHAR(_DeprecationTestCase):

    def test_npy_char_deprecation(self):
        from numpy.core._multiarray_tests import npy_char_deprecation
        self.assert_deprecated(npy_char_deprecation)
        assert_(npy_char_deprecation() == 'S1')
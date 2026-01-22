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
class TestMathAlias(_DeprecationTestCase):

    def test_deprecated_np_math(self):
        self.assert_deprecated(lambda: np.math)

    def test_deprecated_np_lib_math(self):
        self.assert_deprecated(lambda: np.lib.math)
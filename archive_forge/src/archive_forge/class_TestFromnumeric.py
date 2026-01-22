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
class TestFromnumeric(_DeprecationTestCase):

    def test_round_(self):
        self.assert_deprecated(lambda: np.round_(np.array([1.5, 2.5, 3.5])))

    def test_cumproduct(self):
        self.assert_deprecated(lambda: np.cumproduct(np.array([1, 2, 3])))

    def test_product(self):
        self.assert_deprecated(lambda: np.product(np.array([1, 2, 3])))

    def test_sometrue(self):
        self.assert_deprecated(lambda: np.sometrue(np.array([True, False])))

    def test_alltrue(self):
        self.assert_deprecated(lambda: np.alltrue(np.array([True, False])))
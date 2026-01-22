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
class TestGeneratorSum(_DeprecationTestCase):

    def test_generator_sum(self):
        self.assert_deprecated(np.sum, args=((i for i in range(5)),))
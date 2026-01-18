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
def test_type_aliases(self):
    self.assert_deprecated(lambda: np.bool8)
    self.assert_deprecated(lambda: np.int0)
    self.assert_deprecated(lambda: np.uint0)
    self.assert_deprecated(lambda: np.bytes0)
    self.assert_deprecated(lambda: np.str0)
    self.assert_deprecated(lambda: np.object0)
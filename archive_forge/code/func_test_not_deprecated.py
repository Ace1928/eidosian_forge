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
def test_not_deprecated(self, func):
    self.assert_not_deprecated(lambda: func(1))
    self.assert_not_deprecated(lambda: func([0, 1]))
import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
@pytest.mark.parametrize(['dtype1', 'dtype2'], [[np.dtype('V10'), np.dtype('V10')], [np.dtype([('name1', 'i8')]), np.dtype([('name1', np.dtype('i8').newbyteorder())])], [np.dtype('i8,i8'), np.dtype('i8,>i8')], [np.dtype('i8,i8'), np.dtype('i4,i4')]])
def test_valid_void_promotion(self, dtype1, dtype2):
    assert np.promote_types(dtype1, dtype2) == dtype1
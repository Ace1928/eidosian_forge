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
def test_shape_mismatch_error_message(self):
    with pytest.raises(ValueError, match='arg 0 with shape \\(1, 3\\) and arg 2 with shape \\(2,\\)'):
        np.broadcast([[1, 2, 3]], [[4], [5]], [6, 7])
import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
@pytest.mark.parametrize('operation', [operator.le, operator.lt, operator.ge, operator.gt])
def test_richcompare_invalid_dtype_comparison(self, operation):
    with pytest.raises(TypeError):
        operation(np.dtype(np.int32), 7)
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
@pytest.mark.parametrize('DType', [type(np.dtype(t)) for t in np.typecodes['All']] + [np.dtype(rational), np.dtype])
def test_pickle_types(self, DType):
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        roundtrip_DType = pickle.loads(pickle.dumps(DType, proto))
        assert roundtrip_DType is DType
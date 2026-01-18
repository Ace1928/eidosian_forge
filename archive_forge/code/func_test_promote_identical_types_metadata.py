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
@pytest.mark.parametrize('dtype', list(np.typecodes['All']) + ['i,i', '10i', 'S3', 'S100', 'U3', 'U100', rational])
def test_promote_identical_types_metadata(self, dtype):
    metadata = {1: 1}
    dtype = np.dtype(dtype, metadata=metadata)
    res = np.promote_types(dtype, dtype)
    assert res.metadata == dtype.metadata
    dtype = dtype.newbyteorder()
    if dtype.isnative:
        return
    res = np.promote_types(dtype, dtype)
    if dtype.char != 'U':
        assert res.metadata is None
    else:
        assert res.metadata == metadata
    assert res.isnative
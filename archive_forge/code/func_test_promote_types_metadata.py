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
@pytest.mark.slow
@pytest.mark.filterwarnings('ignore:Promotion of numbers:FutureWarning')
@pytest.mark.parametrize(['dtype1', 'dtype2'], itertools.product(list(np.typecodes['All']) + ['i,i', 'S3', 'S100', 'U3', 'U100', rational], repeat=2))
def test_promote_types_metadata(self, dtype1, dtype2):
    """Metadata handling in promotion does not appear formalized
        right now in NumPy. This test should thus be considered to
        document behaviour, rather than test the correct definition of it.

        This test is very ugly, it was useful for rewriting part of the
        promotion, but probably should eventually be replaced/deleted
        (i.e. when metadata handling in promotion is better defined).
        """
    metadata1 = {1: 1}
    metadata2 = {2: 2}
    dtype1 = np.dtype(dtype1, metadata=metadata1)
    dtype2 = np.dtype(dtype2, metadata=metadata2)
    try:
        res = np.promote_types(dtype1, dtype2)
    except TypeError:
        return
    if res.char not in 'USV' or res.names is not None or res.shape != ():
        assert res.metadata is None
    elif res == dtype1:
        assert res is dtype1
    elif res == dtype2:
        if np.promote_types(dtype1, dtype2.kind) == dtype2:
            res.metadata is None
        else:
            res.metadata == metadata2
    else:
        assert res.metadata is None
    dtype1 = dtype1.newbyteorder()
    assert dtype1.metadata == metadata1
    res_bs = np.promote_types(dtype1, dtype2)
    assert res_bs == res
    assert res_bs.metadata == res.metadata
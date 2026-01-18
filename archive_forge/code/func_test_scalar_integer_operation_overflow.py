import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
@pytest.mark.parametrize('dtype', np.typecodes['AllInteger'])
@pytest.mark.parametrize('operation', [lambda min, max: max + max, lambda min, max: min - max, lambda min, max: max * max], ids=['+', '-', '*'])
def test_scalar_integer_operation_overflow(dtype, operation):
    st = np.dtype(dtype).type
    min = st(np.iinfo(dtype).min)
    max = st(np.iinfo(dtype).max)
    with pytest.warns(RuntimeWarning, match='overflow encountered'):
        operation(min, max)
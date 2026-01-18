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
@pytest.mark.parametrize('dtype', np.typecodes['Integer'])
@pytest.mark.parametrize('operation', [lambda min, neg_1: -min, lambda min, neg_1: abs(min), lambda min, neg_1: min * neg_1, pytest.param(lambda min, neg_1: min // neg_1, marks=pytest.mark.skip(reason='broken on some platforms'))], ids=['neg', 'abs', '*', '//'])
def test_scalar_signed_integer_overflow(dtype, operation):
    st = np.dtype(dtype).type
    min = st(np.iinfo(dtype).min)
    neg_1 = st(-1)
    with pytest.warns(RuntimeWarning, match='overflow encountered'):
        operation(min, neg_1)
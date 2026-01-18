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
def test_scalar_comparison_to_none(self):
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', FutureWarning)
        assert_(not np.float32(1) == None)
        assert_(not np.str_('test') == None)
        assert_(not np.datetime64('NaT') == None)
        assert_(np.float32(1) != None)
        assert_(np.str_('test') != None)
        assert_(np.datetime64('NaT') != None)
    assert_(len(w) == 0)
    assert_(np.equal(np.datetime64('NaT'), None))
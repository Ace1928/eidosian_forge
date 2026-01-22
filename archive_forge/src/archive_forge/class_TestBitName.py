import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class TestBitName:

    def test_abstract(self):
        assert_raises(ValueError, np.core.numerictypes.bitname, np.floating)
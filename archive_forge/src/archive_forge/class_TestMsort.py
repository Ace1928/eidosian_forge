import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
class TestMsort:

    def test_simple(self):
        A = np.array([[0.44567325, 0.79115165, 0.5490053], [0.36844147, 0.37325583, 0.96098397], [0.64864341, 0.52929049, 0.39172155]])
        with pytest.warns(DeprecationWarning, match='msort is deprecated'):
            assert_almost_equal(msort(A), np.array([[0.36844147, 0.37325583, 0.39172155], [0.44567325, 0.52929049, 0.5490053], [0.64864341, 0.79115165, 0.96098397]]))
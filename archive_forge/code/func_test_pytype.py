import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_pytype(self):
    r = np.ma.median([[np.inf, np.inf], [np.inf, np.inf]], axis=-1)
    assert_equal(r, np.inf)
import warnings
import platform
import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask
import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
from numpy.testing import suppress_warnings
from scipy.stats import _mstats_basic
def test_winsorization(self):
    data = ma.array([77, 87, 88, 114, 151, 210, 219, 246, 253, 262, 296, 299, 306, 376, 428, 515, 666, 1310, 2611])
    assert_almost_equal(mstats.winsorize(data, (0.2, 0.2)).var(ddof=1), 21551.4, 1)
    assert_almost_equal(mstats.winsorize(data, (0.2, 0.2), (False, False)).var(ddof=1), 11887.3, 1)
    data[5] = masked
    winsorized = mstats.winsorize(data)
    assert_equal(winsorized.mask, data.mask)
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
def test_plotting_positions():
    pos = mstats.plotting_positions(np.arange(3), 0, 0)
    assert_array_almost_equal(pos.data, np.array([0.25, 0.5, 0.75]))
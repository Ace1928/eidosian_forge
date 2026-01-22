from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
class TestAsinhLocator:

    def test_init(self):
        lctr = mticker.AsinhLocator(linear_width=2.718, numticks=19)
        assert lctr.linear_width == 2.718
        assert lctr.numticks == 19
        assert lctr.base == 10

    def test_set_params(self):
        lctr = mticker.AsinhLocator(linear_width=5, numticks=17, symthresh=0.125, base=4, subs=(2.5, 3.25))
        assert lctr.numticks == 17
        assert lctr.symthresh == 0.125
        assert lctr.base == 4
        assert lctr.subs == (2.5, 3.25)
        lctr.set_params(numticks=23)
        assert lctr.numticks == 23
        lctr.set_params(None)
        assert lctr.numticks == 23
        lctr.set_params(symthresh=0.5)
        assert lctr.symthresh == 0.5
        lctr.set_params(symthresh=None)
        assert lctr.symthresh == 0.5
        lctr.set_params(base=7)
        assert lctr.base == 7
        lctr.set_params(base=None)
        assert lctr.base == 7
        lctr.set_params(subs=(2, 4.125))
        assert lctr.subs == (2, 4.125)
        lctr.set_params(subs=None)
        assert lctr.subs == (2, 4.125)
        lctr.set_params(subs=[])
        assert lctr.subs is None

    def test_linear_values(self):
        lctr = mticker.AsinhLocator(linear_width=100, numticks=11, base=0)
        assert_almost_equal(lctr.tick_values(-1, 1), np.arange(-1, 1.01, 0.2))
        assert_almost_equal(lctr.tick_values(-0.1, 0.1), np.arange(-0.1, 0.101, 0.02))
        assert_almost_equal(lctr.tick_values(-0.01, 0.01), np.arange(-0.01, 0.0101, 0.002))

    def test_wide_values(self):
        lctr = mticker.AsinhLocator(linear_width=0.1, numticks=11, base=0)
        assert_almost_equal(lctr.tick_values(-100, 100), [-100, -20, -5, -1, -0.2, 0, 0.2, 1, 5, 20, 100])
        assert_almost_equal(lctr.tick_values(-1000, 1000), [-1000, -100, -20, -3, -0.4, 0, 0.4, 3, 20, 100, 1000])

    def test_near_zero(self):
        """Check that manually injected zero will supersede nearby tick"""
        lctr = mticker.AsinhLocator(linear_width=100, numticks=3, base=0)
        assert_almost_equal(lctr.tick_values(-1.1, 0.9), [-1.0, 0.0, 0.9])

    def test_fallback(self):
        lctr = mticker.AsinhLocator(1.0, numticks=11)
        assert_almost_equal(lctr.tick_values(101, 102), np.arange(101, 102.01, 0.1))

    def test_symmetrizing(self):
        lctr = mticker.AsinhLocator(linear_width=1, numticks=3, symthresh=0.25, base=0)
        lctr.create_dummy_axis()
        lctr.axis.set_view_interval(-1, 2)
        assert_almost_equal(lctr(), [-1, 0, 2])
        lctr.axis.set_view_interval(-1, 0.9)
        assert_almost_equal(lctr(), [-1, 0, 1])
        lctr.axis.set_view_interval(-0.85, 1.05)
        assert_almost_equal(lctr(), [-1, 0, 1])
        lctr.axis.set_view_interval(1, 1.1)
        assert_almost_equal(lctr(), [1, 1.05, 1.1])

    def test_base_rounding(self):
        lctr10 = mticker.AsinhLocator(linear_width=1, numticks=8, base=10, subs=(1, 3, 5))
        assert_almost_equal(lctr10.tick_values(-110, 110), [-500, -300, -100, -50, -30, -10, -5, -3, -1, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 50, 100, 300, 500])
        lctr5 = mticker.AsinhLocator(linear_width=1, numticks=20, base=5)
        assert_almost_equal(lctr5.tick_values(-1050, 1050), [-625, -125, -25, -5, -1, -0.2, 0, 0.2, 1, 5, 25, 125, 625])
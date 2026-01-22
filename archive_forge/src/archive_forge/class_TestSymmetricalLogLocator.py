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
class TestSymmetricalLogLocator:

    def test_set_params(self):
        """
        Create symmetrical log locator with default subs =[1.0] numticks = 15,
        and change it to something else.
        See if change was successful.
        Should not exception.
        """
        sym = mticker.SymmetricalLogLocator(base=10, linthresh=1)
        sym.set_params(subs=[2.0], numticks=8)
        assert sym._subs == [2.0]
        assert sym.numticks == 8

    @pytest.mark.parametrize('vmin, vmax, expected', [(0, 1, [0, 1]), (-1, 1, [-1, 0, 1])])
    def test_values(self, vmin, vmax, expected):
        sym = mticker.SymmetricalLogLocator(base=10, linthresh=1)
        ticks = sym.tick_values(vmin=vmin, vmax=vmax)
        assert_array_equal(ticks, expected)

    def test_subs(self):
        sym = mticker.SymmetricalLogLocator(base=10, linthresh=1, subs=[2.0, 4.0])
        sym.create_dummy_axis()
        sym.axis.set_view_interval(-10, 10)
        assert (sym() == [-20.0, -40.0, -2.0, -4.0, 0.0, 2.0, 4.0, 20.0, 40.0]).all()

    def test_extending(self):
        sym = mticker.SymmetricalLogLocator(base=10, linthresh=1)
        sym.create_dummy_axis()
        sym.axis.set_view_interval(8, 9)
        assert (sym() == [1.0]).all()
        sym.axis.set_view_interval(8, 12)
        assert (sym() == [1.0, 10.0]).all()
        assert sym.view_limits(10, 10) == (1, 100)
        assert sym.view_limits(-10, -10) == (-100, -1)
        assert sym.view_limits(0, 0) == (-0.001, 0.001)
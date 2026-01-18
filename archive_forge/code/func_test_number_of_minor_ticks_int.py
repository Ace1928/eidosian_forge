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
@pytest.mark.parametrize('use_rcparam', [False, True])
@pytest.mark.parametrize('n, lim, ref', [(2, (0, 4), [0.5, 1.5, 2.5, 3.5]), (4, (0, 2), [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]), (10, (0, 1), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])])
def test_number_of_minor_ticks_int(self, n, lim, ref, use_rcparam):
    if use_rcparam:
        context = {'xtick.minor.ndivs': n, 'ytick.minor.ndivs': n}
        kwargs = {}
    else:
        context = {}
        kwargs = {'n': n}
    with mpl.rc_context(context):
        fig, ax = plt.subplots()
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(**kwargs))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(**kwargs))
        assert_almost_equal(ax.xaxis.get_ticklocs(minor=True), ref)
        assert_almost_equal(ax.yaxis.get_ticklocs(minor=True), ref)
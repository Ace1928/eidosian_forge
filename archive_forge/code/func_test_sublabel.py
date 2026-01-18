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
@mpl.style.context('default')
def test_sublabel(self):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=[]))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_major_formatter(mticker.LogFormatter(labelOnlyBase=True))
    ax.xaxis.set_minor_formatter(mticker.LogFormatter(labelOnlyBase=False))
    ax.set_xlim(1, 10000.0)
    fmt = ax.xaxis.get_major_formatter()
    fmt.set_locs(ax.xaxis.get_majorticklocs())
    show_major_labels = [fmt(x) != '' for x in ax.xaxis.get_majorticklocs()]
    assert np.all(show_major_labels)
    self._sub_labels(ax.xaxis, subs=[])
    ax.set_xlim(1, 800)
    self._sub_labels(ax.xaxis, subs=[])
    ax.set_xlim(1, 80)
    self._sub_labels(ax.xaxis, subs=[])
    ax.set_xlim(1, 8)
    self._sub_labels(ax.xaxis, subs=[2, 3, 4, 6])
    ax.set_xlim(0.5, 0.9)
    self._sub_labels(ax.xaxis, subs=np.arange(2, 10, dtype=int))
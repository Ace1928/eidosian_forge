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
@pytest.mark.parametrize('remove_overlapping_locs, expected_num', ((True, 6), (None, 6), (False, 9)))
def test_remove_overlap(remove_overlapping_locs, expected_num):
    t = np.arange('2018-11-03', '2018-11-06', dtype='datetime64')
    x = np.ones(len(t))
    fig, ax = plt.subplots()
    ax.plot(t, x)
    ax.xaxis.set_major_locator(mpl.dates.DayLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('\n%a'))
    ax.xaxis.set_minor_locator(mpl.dates.HourLocator((0, 6, 12, 18)))
    ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax.xaxis.get_minor_ticks(15)
    if remove_overlapping_locs is not None:
        ax.xaxis.remove_overlapping_locs = remove_overlapping_locs
    current = ax.xaxis.remove_overlapping_locs
    assert current == ax.xaxis.get_remove_overlapping_locs()
    plt.setp(ax.xaxis, remove_overlapping_locs=current)
    new = ax.xaxis.remove_overlapping_locs
    assert new == ax.xaxis.remove_overlapping_locs
    assert len(ax.xaxis.get_minorticklocs()) == expected_num
    assert len(ax.xaxis.get_minor_ticks()) == expected_num
    assert len(ax.xaxis.get_minorticklabels()) == expected_num
    assert len(ax.xaxis.get_minorticklines()) == expected_num * 2
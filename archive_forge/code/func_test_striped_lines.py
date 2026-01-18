from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@pytest.mark.parametrize('gapcolor', ['orange', ['r', 'k']])
@check_figures_equal(extensions=['png'])
@mpl.rc_context({'lines.linewidth': 20})
def test_striped_lines(fig_test, fig_ref, gapcolor):
    ax_test = fig_test.add_subplot(111)
    ax_ref = fig_ref.add_subplot(111)
    for ax in [ax_test, ax_ref]:
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1)
    x = range(1, 6)
    linestyles = [':', '-', '--']
    ax_test.vlines(x, 0, 1, linestyle=linestyles, gapcolor=gapcolor, alpha=0.5)
    if isinstance(gapcolor, str):
        gapcolor = [gapcolor]
    for x, gcol, ls in zip(x, itertools.cycle(gapcolor), itertools.cycle(linestyles)):
        ax_ref.axvline(x, 0, 1, linestyle=ls, gapcolor=gcol, alpha=0.5)
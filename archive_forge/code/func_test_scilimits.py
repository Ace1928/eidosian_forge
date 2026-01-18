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
@pytest.mark.parametrize('sci_type, scilimits, lim, orderOfMag, fewticks', scilimits_data)
def test_scilimits(self, sci_type, scilimits, lim, orderOfMag, fewticks):
    tmp_form = mticker.ScalarFormatter()
    tmp_form.set_scientific(sci_type)
    tmp_form.set_powerlimits(scilimits)
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(tmp_form)
    ax.set_ylim(*lim)
    if fewticks:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    tmp_form.set_locs(ax.yaxis.get_majorticklocs())
    assert orderOfMag == tmp_form.orderOfMagnitude
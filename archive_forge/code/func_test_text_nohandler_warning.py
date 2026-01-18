import collections
import platform
from unittest import mock
import warnings
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import matplotlib.legend as mlegend
from matplotlib import _api, rc_context
from matplotlib.font_manager import FontProperties
def test_text_nohandler_warning():
    """Test that Text artists with labels raise a warning"""
    fig, ax = plt.subplots()
    ax.text(x=0, y=0, s='text', label='label')
    with pytest.warns(UserWarning) as record:
        ax.legend()
    assert len(record) == 1
    f, ax = plt.subplots()
    ax.pcolormesh(np.random.uniform(0, 1, (10, 10)))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ax.get_legend_handles_labels()
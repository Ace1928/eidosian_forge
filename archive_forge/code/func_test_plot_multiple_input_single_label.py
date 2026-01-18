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
@pytest.mark.parametrize('label', ['one', 1, int])
def test_plot_multiple_input_single_label(label):
    x = [1, 2, 3]
    y = [[1, 2], [2, 5], [4, 9]]
    fig, ax = plt.subplots()
    ax.plot(x, y, label=label)
    leg = ax.legend()
    legend_texts = [entry.get_text() for entry in leg.get_texts()]
    assert legend_texts == [str(label)] * 2
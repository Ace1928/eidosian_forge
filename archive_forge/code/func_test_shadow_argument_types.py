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
@image_comparison(['shadow_argument_types.png'], remove_text=True, style='mpl20')
def test_shadow_argument_types():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], label='test')
    legs = (ax.legend(loc='upper left', shadow=True), ax.legend(loc='upper right', shadow=False), ax.legend(loc='center left', shadow={'color': 'red', 'alpha': 0.1}), ax.legend(loc='center right', shadow={'color': (0.1, 0.2, 0.5), 'oy': -5}), ax.legend(loc='lower left', shadow={'color': 'tab:cyan', 'ox': 10}))
    for l in legs:
        ax.add_artist(l)
    ax.legend(loc='lower right')
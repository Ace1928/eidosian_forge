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
@image_comparison(['scatter_rc3', 'scatter_rc1'], remove_text=True)
def test_rc():
    plt.figure()
    ax = plt.subplot(121)
    ax.scatter(np.arange(10), np.arange(10, 0, -1), label='three')
    ax.legend(loc='center left', bbox_to_anchor=[1.0, 0.5], title='My legend')
    mpl.rcParams['legend.scatterpoints'] = 1
    plt.figure()
    ax = plt.subplot(121)
    ax.scatter(np.arange(10), np.arange(10, 0, -1), label='one')
    ax.legend(loc='center left', bbox_to_anchor=[1.0, 0.5], title='My legend')
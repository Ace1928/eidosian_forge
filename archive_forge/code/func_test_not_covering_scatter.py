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
@image_comparison(['not_covering_scatter.png'])
def test_not_covering_scatter():
    colors = ['b', 'g', 'r']
    for n in range(3):
        plt.scatter([n], [n], color=colors[n])
    plt.legend(['foo', 'foo', 'foo'], loc='best')
    plt.gca().set_xlim(-0.5, 2.2)
    plt.gca().set_ylim(-0.5, 2.2)
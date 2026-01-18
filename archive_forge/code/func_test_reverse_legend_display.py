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
@check_figures_equal(extensions=['png'])
def test_reverse_legend_display(fig_test, fig_ref):
    """Check that the rendered legend entries are reversed"""
    ax = fig_test.subplots()
    ax.plot([1], 'ro', label='first')
    ax.plot([2], 'bx', label='second')
    ax.legend(reverse=True)
    ax = fig_ref.subplots()
    ax.plot([2], 'bx', label='second')
    ax.plot([1], 'ro', label='first')
    ax.legend()
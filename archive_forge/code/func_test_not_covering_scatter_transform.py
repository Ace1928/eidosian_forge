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
@image_comparison(['not_covering_scatter_transform.png'])
def test_not_covering_scatter_transform():
    offset = mtransforms.Affine2D().translate(-20, 20)
    x = np.linspace(0, 30, 1000)
    plt.plot(x, x)
    plt.scatter([20], [10], transform=offset + plt.gca().transData)
    plt.legend(['foo', 'bar'], loc='best')
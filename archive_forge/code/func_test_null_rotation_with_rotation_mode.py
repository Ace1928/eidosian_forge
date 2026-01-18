from datetime import datetime
import io
import warnings
import numpy as np
from numpy.testing import assert_almost_equal
from packaging.version import parse as parse_version
import pyparsing
import pytest
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib.text import Text, Annotation, OffsetFrom
@pytest.mark.parametrize('ha', ['center', 'right', 'left'])
@pytest.mark.parametrize('va', ['center', 'top', 'bottom', 'baseline', 'center_baseline'])
def test_null_rotation_with_rotation_mode(ha, va):
    fig, ax = plt.subplots()
    kw = dict(rotation=0, va=va, ha=ha)
    t0 = ax.text(0.5, 0.5, 'test', rotation_mode='anchor', **kw)
    t1 = ax.text(0.5, 0.5, 'test', rotation_mode='default', **kw)
    fig.canvas.draw()
    assert_almost_equal(t0.get_window_extent(fig.canvas.renderer).get_points(), t1.get_window_extent(fig.canvas.renderer).get_points())
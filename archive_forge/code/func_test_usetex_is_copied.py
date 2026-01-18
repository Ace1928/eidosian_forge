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
@needs_usetex
def test_usetex_is_copied():
    fig = plt.figure()
    plt.rcParams['text.usetex'] = False
    ax1 = fig.add_subplot(121)
    plt.rcParams['text.usetex'] = True
    ax2 = fig.add_subplot(122)
    fig.canvas.draw()
    for ax, usetex in [(ax1, False), (ax2, True)]:
        for t in ax.xaxis.majorTicks:
            assert t.label1.get_usetex() == usetex
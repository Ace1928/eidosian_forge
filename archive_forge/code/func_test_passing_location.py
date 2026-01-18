import platform
import numpy as np
import pytest
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import rc_context
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import (
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FixedLocator, LogFormatter, StrMethodFormatter
from matplotlib.testing.decorators import check_figures_equal
@check_figures_equal(extensions=['png'])
def test_passing_location(fig_ref, fig_test):
    ax_ref = fig_ref.add_subplot()
    im = ax_ref.imshow([[0, 1], [2, 3]])
    ax_ref.figure.colorbar(im, cax=ax_ref.inset_axes([0, 1.05, 1, 0.05]), orientation='horizontal', ticklocation='top')
    ax_test = fig_test.add_subplot()
    im = ax_test.imshow([[0, 1], [2, 3]])
    ax_test.figure.colorbar(im, cax=ax_test.inset_axes([0, 1.05, 1, 0.05]), location='top')
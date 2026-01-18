from itertools import product
import io
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cbook
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.testing.decorators import (
from mpl_toolkits.axes_grid1 import (
from mpl_toolkits.axes_grid1.anchored_artists import (
from mpl_toolkits.axes_grid1.axes_divider import (
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
from mpl_toolkits.axes_grid1.inset_locator import (
import mpl_toolkits.axes_grid1.mpl_axes
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
@image_comparison(['twin_axes_empty_and_removed'], extensions=['png'], tol=1, style=('classic', '_classic_test_patch'))
def test_twin_axes_empty_and_removed():
    mpl.rcParams.update({'font.size': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
    generators = ['twinx', 'twiny', 'twin']
    modifiers = ['', 'host invisible', 'twin removed', 'twin invisible', 'twin removed\nhost invisible']
    h = host_subplot(len(modifiers) + 1, len(generators), 2)
    h.text(0.5, 0.5, 'host_subplot', horizontalalignment='center', verticalalignment='center')
    for i, (mod, gen) in enumerate(product(modifiers, generators), len(generators) + 1):
        h = host_subplot(len(modifiers) + 1, len(generators), i)
        t = getattr(h, gen)()
        if 'twin invisible' in mod:
            t.axis[:].set_visible(False)
        if 'twin removed' in mod:
            t.remove()
        if 'host invisible' in mod:
            h.axis[:].set_visible(False)
        h.text(0.5, 0.5, gen + ('\n' + mod if mod else ''), horizontalalignment='center', verticalalignment='center')
    plt.subplots_adjust(wspace=0.5, hspace=1)
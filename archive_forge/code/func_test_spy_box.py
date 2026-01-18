from contextlib import ExitStack
from copy import copy
import functools
import io
import os
from pathlib import Path
import platform
import sys
import urllib.request
import numpy as np
from numpy.testing import assert_array_equal
from PIL import Image
import matplotlib as mpl
from matplotlib import (
from matplotlib.image import (AxesImage, BboxImage, FigureImage,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.transforms import Bbox, Affine2D, TransformedBbox
import matplotlib.ticker as mticker
import pytest
@check_figures_equal()
def test_spy_box(fig_test, fig_ref):
    ax_test = fig_test.subplots(1, 3)
    ax_ref = fig_ref.subplots(1, 3)
    plot_data = ([[1, 1], [1, 1]], [[0, 0], [0, 0]], [[0, 1], [1, 0]])
    plot_titles = ['ones', 'zeros', 'mixed']
    for i, (z, title) in enumerate(zip(plot_data, plot_titles)):
        ax_test[i].set_title(title)
        ax_test[i].spy(z)
        ax_ref[i].set_title(title)
        ax_ref[i].imshow(z, interpolation='nearest', aspect='equal', origin='upper', cmap='Greys', vmin=0, vmax=1)
        ax_ref[i].set_xlim(-0.5, 1.5)
        ax_ref[i].set_ylim(1.5, -0.5)
        ax_ref[i].xaxis.tick_top()
        ax_ref[i].title.set_y(1.05)
        ax_ref[i].xaxis.set_ticks_position('both')
        ax_ref[i].xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        ax_ref[i].yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
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
@check_figures_equal(extensions=['png'])
def test_norm_change(fig_test, fig_ref):
    data = np.full((5, 5), 1, dtype=np.float64)
    data[0:2, :] = -1
    masked_data = np.ma.array(data, mask=False)
    masked_data.mask[0:2, 0:2] = True
    cmap = mpl.colormaps['viridis'].with_extremes(under='w')
    ax = fig_test.subplots()
    im = ax.imshow(data, norm=colors.LogNorm(vmin=0.5, vmax=1), extent=(0, 5, 0, 5), interpolation='nearest', cmap=cmap)
    im.set_norm(colors.Normalize(vmin=-2, vmax=2))
    im = ax.imshow(masked_data, norm=colors.LogNorm(vmin=0.5, vmax=1), extent=(5, 10, 5, 10), interpolation='nearest', cmap=cmap)
    im.set_norm(colors.Normalize(vmin=-2, vmax=2))
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax = fig_ref.subplots()
    ax.imshow(data, norm=colors.Normalize(vmin=-2, vmax=2), extent=(0, 5, 0, 5), interpolation='nearest', cmap=cmap)
    ax.imshow(masked_data, norm=colors.Normalize(vmin=-2, vmax=2), extent=(5, 10, 5, 10), interpolation='nearest', cmap=cmap)
    ax.set(xlim=(0, 10), ylim=(0, 10))
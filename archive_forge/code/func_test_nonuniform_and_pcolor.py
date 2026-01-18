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
@image_comparison(['nonuniform_and_pcolor.png'], style='mpl20')
def test_nonuniform_and_pcolor():
    axs = plt.figure(figsize=(3, 3)).subplots(3, sharex=True, sharey=True)
    for ax, interpolation in zip(axs, ['nearest', 'bilinear']):
        im = NonUniformImage(ax, interpolation=interpolation)
        im.set_data(np.arange(3) ** 2, np.arange(3) ** 2, np.arange(9).reshape((3, 3)))
        ax.add_image(im)
    axs[2].pcolorfast(np.arange(4) ** 2, np.arange(4) ** 2, np.arange(9).reshape((3, 3)))
    for ax in axs:
        ax.set_axis_off()
        ax.set(xlim=(0, 10))
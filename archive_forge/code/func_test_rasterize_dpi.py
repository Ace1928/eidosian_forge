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
@image_comparison(['rasterize_10dpi'], extensions=['pdf', 'svg'], remove_text=True, style='mpl20')
def test_rasterize_dpi():
    img = np.asarray([[1, 2], [3, 4]])
    fig, axs = plt.subplots(1, 3, figsize=(3, 1))
    axs[0].imshow(img)
    axs[1].plot([0, 1], [0, 1], linewidth=20.0, rasterized=True)
    axs[1].set(xlim=(0, 1), ylim=(-1, 2))
    axs[2].plot([0, 1], [0, 1], linewidth=20.0)
    axs[2].set(xlim=(0, 1), ylim=(-1, 2))
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    rcParams['savefig.dpi'] = 10
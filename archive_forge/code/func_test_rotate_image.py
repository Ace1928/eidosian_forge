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
@image_comparison(['rotate_image'], remove_text=True)
def test_rotate_image():
    delta = 0.25
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-(X ** 2 + Y ** 2) / 2) / (2 * np.pi)
    Z2 = np.exp(-(((X - 1) / 1.5) ** 2 + ((Y - 1) / 0.5) ** 2) / 2) / (2 * np.pi * 0.5 * 1.5)
    Z = Z2 - Z1
    fig, ax1 = plt.subplots(1, 1)
    im1 = ax1.imshow(Z, interpolation='none', cmap='viridis', origin='lower', extent=[-2, 4, -3, 2], clip_on=True)
    trans_data2 = Affine2D().rotate_deg(30) + ax1.transData
    im1.set_transform(trans_data2)
    x1, x2, y1, y2 = im1.get_extent()
    ax1.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r--', lw=3, transform=trans_data2)
    ax1.set_xlim(2, 5)
    ax1.set_ylim(0, 4)
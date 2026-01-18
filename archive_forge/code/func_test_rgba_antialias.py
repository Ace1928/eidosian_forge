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
@image_comparison(['rgba_antialias.png'], style='mpl20', remove_text=True, tol=0.007 if platform.machine() in ('aarch64', 'ppc64le', 's390x') else 0)
def test_rgba_antialias():
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.5), sharex=False, sharey=False, constrained_layout=True)
    N = 250
    aa = np.ones((N, N))
    aa[::2, :] = -1
    x = np.arange(N) / N - 0.5
    y = np.arange(N) / N - 0.5
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    f0 = 10
    k = 75
    a = np.sin(np.pi * 2 * (f0 * R + k * R ** 2 / 2))
    a[:int(N / 2), :][R[:int(N / 2), :] < 0.4] = -1
    a[:int(N / 2), :][R[:int(N / 2), :] < 0.3] = 1
    aa[:, int(N / 2):] = a[:, int(N / 2):]
    aa[20:50, 20:50] = np.nan
    aa[70:90, 70:90] = 1000000.0
    aa[70:90, 20:30] = -1000000.0
    aa[70:90, 195:215] = 1000000.0
    aa[20:30, 195:215] = -1000000.0
    cmap = copy(plt.cm.RdBu_r)
    cmap.set_over('yellow')
    cmap.set_under('cyan')
    axs = axs.flatten()
    axs[0].imshow(aa, interpolation='nearest', cmap=cmap, vmin=-1.2, vmax=1.2)
    axs[0].set_xlim([N / 2 - 25, N / 2 + 25])
    axs[0].set_ylim([N / 2 + 50, N / 2 - 10])
    axs[1].imshow(aa, interpolation='nearest', cmap=cmap, vmin=-1.2, vmax=1.2)
    axs[2].imshow(aa, interpolation='antialiased', interpolation_stage='data', cmap=cmap, vmin=-1.2, vmax=1.2)
    axs[3].imshow(aa, interpolation='antialiased', interpolation_stage='rgba', cmap=cmap, vmin=-1.2, vmax=1.2)
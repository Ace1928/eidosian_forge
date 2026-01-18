import copy
import itertools
import unittest.mock
from packaging.version import parse as parse_version
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
import base64
from numpy.testing import assert_array_equal, assert_array_almost_equal
from matplotlib import cbook, cm
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.rcsetup import cycler
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@check_figures_equal()
def test_norm_update_figs(fig_test, fig_ref):
    ax_ref = fig_ref.add_subplot()
    ax_test = fig_test.add_subplot()
    z = np.arange(100).reshape((10, 10))
    ax_ref.imshow(z, norm=mcolors.Normalize(10, 90))
    norm = mcolors.Normalize(0, 1)
    ax_test.imshow(z, norm=norm)
    fig_test.canvas.draw()
    norm.vmin, norm.vmax = (10, 90)
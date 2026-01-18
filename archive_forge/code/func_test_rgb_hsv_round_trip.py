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
def test_rgb_hsv_round_trip():
    for a_shape in [(500, 500, 3), (500, 3), (1, 3), (3,)]:
        np.random.seed(0)
        tt = np.random.random(a_shape)
        assert_array_almost_equal(tt, mcolors.hsv_to_rgb(mcolors.rgb_to_hsv(tt)))
        assert_array_almost_equal(tt, mcolors.rgb_to_hsv(mcolors.hsv_to_rgb(tt)))
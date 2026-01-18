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
def test_to_rgba_array_accepts_color_alpha_tuple_with_multiple_colors():
    color_array = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
    assert_array_equal(mcolors.to_rgba_array((color_array, 0.2)), [[1.0, 1.0, 1.0, 0.2], [0.0, 0.0, 1.0, 0.2]])
    color_sequence = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    assert_array_equal(mcolors.to_rgba_array((color_sequence, 0.4)), [[1.0, 1.0, 1.0, 0.4], [0.0, 0.0, 1.0, 0.4]])
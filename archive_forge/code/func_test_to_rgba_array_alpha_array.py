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
def test_to_rgba_array_alpha_array():
    with pytest.raises(ValueError, match='The number of colors must match'):
        mcolors.to_rgba_array(np.ones((5, 3), float), alpha=np.ones((2,)))
    alpha = [0.5, 0.6]
    c = mcolors.to_rgba_array(np.ones((2, 3), float), alpha=alpha)
    assert_array_equal(c[:, 3], alpha)
    c = mcolors.to_rgba_array(['r', 'g'], alpha=alpha)
    assert_array_equal(c[:, 3], alpha)
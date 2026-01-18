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
def test_to_rgba_array_single_str():
    assert_array_equal(mcolors.to_rgba_array('red'), [(1, 0, 0, 1)])
    with pytest.raises(ValueError, match="'rgb' is not a valid color value."):
        array = mcolors.to_rgba_array('rgb')
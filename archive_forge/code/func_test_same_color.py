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
def test_same_color():
    assert mcolors.same_color('k', (0, 0, 0))
    assert not mcolors.same_color('w', (1, 1, 0))
    assert mcolors.same_color(['red', 'blue'], ['r', 'b'])
    assert mcolors.same_color('none', 'none')
    assert not mcolors.same_color('none', 'red')
    with pytest.raises(ValueError):
        mcolors.same_color(['r', 'g', 'b'], ['r'])
    with pytest.raises(ValueError):
        mcolors.same_color(['red', 'green'], 'none')
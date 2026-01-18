import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
import sys
@mpl.style.context('default')
def test_patch_color_none():
    c = plt.Circle((0, 0), 1, facecolor='none', alpha=1)
    assert c.get_facecolor()[0] == 0
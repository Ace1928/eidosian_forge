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
def test_quantitynd():
    q = QuantityND([1, 2], 'm')
    q0, q1 = q[:]
    assert np.all(q.v == np.asarray([1, 2]))
    assert q.units == 'm'
    assert np.all((q0 + q1).v == np.asarray([3]))
    assert (q0 * q1).units == 'm*m'
    assert (q1 / q0).units == 'm/(m)'
    with pytest.raises(ValueError):
        q0 + QuantityND(1, 's')
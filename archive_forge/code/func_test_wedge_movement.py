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
def test_wedge_movement():
    param_dict = {'center': ((0, 0), (1, 1), 'set_center'), 'r': (5, 8, 'set_radius'), 'width': (2, 3, 'set_width'), 'theta1': (0, 30, 'set_theta1'), 'theta2': (45, 50, 'set_theta2')}
    init_args = {k: v[0] for k, v in param_dict.items()}
    w = mpatches.Wedge(**init_args)
    for attr, (old_v, new_v, func) in param_dict.items():
        assert getattr(w, attr) == old_v
        getattr(w, func)(new_v)
        assert getattr(w, attr) == new_v
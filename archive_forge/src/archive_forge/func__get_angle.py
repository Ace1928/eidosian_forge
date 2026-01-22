import logging
from types import SimpleNamespace
import numpy as np
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib import _docstring
def _get_angle(a, r):
    if a is None:
        return None
    else:
        return a + r
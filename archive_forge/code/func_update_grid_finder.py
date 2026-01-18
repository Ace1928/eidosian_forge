import functools
from itertools import chain
import numpy as np
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, IdentityTransform
from .axislines import (
from .axis_artist import AxisArtist
from .grid_finder import GridFinder
def update_grid_finder(self, aux_trans=None, **kwargs):
    if aux_trans is not None:
        self.grid_finder.update_transform(aux_trans)
    self.grid_finder.update(**kwargs)
    self._old_limits = None
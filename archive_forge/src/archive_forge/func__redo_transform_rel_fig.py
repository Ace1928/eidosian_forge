from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
def _redo_transform_rel_fig(self, bbox=None):
    """
        Make the transSubfigure bbox relative to Figure transform.

        Parameters
        ----------
        bbox : bbox or None
            If not None, then the bbox is used for relative bounding box.
            Otherwise, it is calculated from the subplotspec.
        """
    if bbox is not None:
        self.bbox_relative.p0 = bbox.p0
        self.bbox_relative.p1 = bbox.p1
        return
    gs = self._subplotspec.get_gridspec()
    wr = np.asarray(gs.get_width_ratios())
    hr = np.asarray(gs.get_height_ratios())
    dx = wr[self._subplotspec.colspan].sum() / wr.sum()
    dy = hr[self._subplotspec.rowspan].sum() / hr.sum()
    x0 = wr[:self._subplotspec.colspan.start].sum() / wr.sum()
    y0 = 1 - hr[:self._subplotspec.rowspan.stop].sum() / hr.sum()
    self.bbox_relative.p0 = (x0, y0)
    self.bbox_relative.p1 = (x0 + dx, y0 + dy)
import math
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.axes import Axes
import matplotlib.axis as maxis
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.spines import Spine
class PolarAffine(mtransforms.Affine2DBase):
    """
    The affine part of the polar projection.

    Scales the output so that maximum radius rests on the edge of the axes
    circle and the origin is mapped to (0.5, 0.5). The transform applied is
    the same to x and y components and given by:

    .. math::

        x_{1} = 0.5 \\left [ \\frac{x_{0}}{(r_{\\max} - r_{\\min})} + 1 \\right ]

    :math:`r_{\\min}, r_{\\max}` are the minimum and maximum radial limits after
    any scaling (e.g. log scaling) has been removed.
    """

    def __init__(self, scale_transform, limits):
        """
        Parameters
        ----------
        scale_transform : `~matplotlib.transforms.Transform`
            Scaling transform for the data. This is used to remove any scaling
            from the radial view limits.
        limits : `~matplotlib.transforms.BboxBase`
            View limits of the data. The only part of its bounds that is used
            is the y limits (for the radius limits).
        """
        super().__init__()
        self._scale_transform = scale_transform
        self._limits = limits
        self.set_children(scale_transform, limits)
        self._mtx = None
    __str__ = mtransforms._make_str_method('_scale_transform', '_limits')

    def get_matrix(self):
        if self._invalid:
            limits_scaled = self._limits.transformed(self._scale_transform)
            yscale = limits_scaled.ymax - limits_scaled.ymin
            affine = mtransforms.Affine2D().scale(0.5 / yscale).translate(0.5, 0.5)
            self._mtx = affine.get_matrix()
            self._inverted = None
            self._invalid = 0
        return self._mtx
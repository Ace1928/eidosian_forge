from collections import defaultdict
import functools
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d
@_preprocess_data()
def tricontour(self, *args, extend3d=False, stride=5, zdir='z', offset=None, **kwargs):
    """
        Create a 3D contour plot.

        .. note::
            This method currently produces incorrect output due to a
            longstanding bug in 3D PolyCollection rendering.

        Parameters
        ----------
        X, Y, Z : array-like
            Input data. See `.Axes.tricontour` for supported data shapes.
        extend3d : bool, default: False
            Whether to extend contour in 3D.
        stride : int
            Step size for extending contour.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : float, optional
            If specified, plot a projection of the contour lines at this
            position in a plane normal to *zdir*.
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        *args, **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.tricontour`.

        Returns
        -------
        matplotlib.tri._tricontour.TriContourSet
        """
    had_data = self.has_data()
    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
    X = tri.x
    Y = tri.y
    if 'Z' in kwargs:
        Z = kwargs.pop('Z')
    else:
        Z, *args = args
    jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
    tri = Triangulation(jX, jY, tri.triangles, tri.mask)
    cset = super().tricontour(tri, jZ, *args, **kwargs)
    self.add_contour_set(cset, extend3d, stride, zdir, offset)
    self.auto_scale_xyz(X, Y, Z, had_data)
    return cset
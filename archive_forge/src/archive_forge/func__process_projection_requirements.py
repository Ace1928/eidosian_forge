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
def _process_projection_requirements(self, *, axes_class=None, polar=False, projection=None, **kwargs):
    """
        Handle the args/kwargs to add_axes/add_subplot/gca, returning::

            (axes_proj_class, proj_class_kwargs)

        which can be used for new Axes initialization/identification.
        """
    if axes_class is not None:
        if polar or projection is not None:
            raise ValueError("Cannot combine 'axes_class' and 'projection' or 'polar'")
        projection_class = axes_class
    else:
        if polar:
            if projection is not None and projection != 'polar':
                raise ValueError(f'polar={polar}, yet projection={projection!r}. Only one of these arguments should be supplied.')
            projection = 'polar'
        if isinstance(projection, str) or projection is None:
            projection_class = projections.get_projection_class(projection)
        elif hasattr(projection, '_as_mpl_axes'):
            projection_class, extra_kwargs = projection._as_mpl_axes()
            kwargs.update(**extra_kwargs)
        else:
            raise TypeError(f'projection must be a string, None or implement a _as_mpl_axes method, not {projection!r}')
    return (projection_class, kwargs)
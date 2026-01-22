import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
from .bezier import (
from .path import Path
from ._enums import JoinStyle, CapStyle
class ConnectionPatch(FancyArrowPatch):
    """A patch that connects two points (possibly in different axes)."""

    def __str__(self):
        return 'ConnectionPatch((%g, %g), (%g, %g))' % (self.xy1[0], self.xy1[1], self.xy2[0], self.xy2[1])

    @_docstring.dedent_interpd
    def __init__(self, xyA, xyB, coordsA, coordsB=None, *, axesA=None, axesB=None, arrowstyle='-', connectionstyle='arc3', patchA=None, patchB=None, shrinkA=0.0, shrinkB=0.0, mutation_scale=10.0, mutation_aspect=None, clip_on=False, **kwargs):
        """
        Connect point *xyA* in *coordsA* with point *xyB* in *coordsB*.

        Valid keys are

        ===============  ======================================================
        Key              Description
        ===============  ======================================================
        arrowstyle       the arrow style
        connectionstyle  the connection style
        relpos           default is (0.5, 0.5)
        patchA           default is bounding box of the text
        patchB           default is None
        shrinkA          default is 2 points
        shrinkB          default is 2 points
        mutation_scale   default is text size (in points)
        mutation_aspect  default is 1.
        ?                any key for `matplotlib.patches.PathPatch`
        ===============  ======================================================

        *coordsA* and *coordsB* are strings that indicate the
        coordinates of *xyA* and *xyB*.

        ==================== ==================================================
        Property             Description
        ==================== ==================================================
        'figure points'      points from the lower left corner of the figure
        'figure pixels'      pixels from the lower left corner of the figure
        'figure fraction'    0, 0 is lower left of figure and 1, 1 is upper
                             right
        'subfigure points'   points from the lower left corner of the subfigure
        'subfigure pixels'   pixels from the lower left corner of the subfigure
        'subfigure fraction' fraction of the subfigure, 0, 0 is lower left.
        'axes points'        points from lower left corner of axes
        'axes pixels'        pixels from lower left corner of axes
        'axes fraction'      0, 0 is lower left of axes and 1, 1 is upper right
        'data'               use the coordinate system of the object being
                             annotated (default)
        'offset points'      offset (in points) from the *xy* value
        'polar'              you can specify *theta*, *r* for the annotation,
                             even in cartesian plots.  Note that if you are
                             using a polar axes, you do not need to specify
                             polar for the coordinate system since that is the
                             native "data" coordinate system.
        ==================== ==================================================

        Alternatively they can be set to any valid
        `~matplotlib.transforms.Transform`.

        Note that 'subfigure pixels' and 'figure pixels' are the same
        for the parent figure, so users who want code that is usable in
        a subfigure can use 'subfigure pixels'.

        .. note::

           Using `ConnectionPatch` across two `~.axes.Axes` instances
           is not directly compatible with :ref:`constrained layout
           <constrainedlayout_guide>`. Add the artist
           directly to the `.Figure` instead of adding it to a specific Axes,
           or exclude it from the layout using ``con.set_in_layout(False)``.

           .. code-block:: default

              fig, ax = plt.subplots(1, 2, constrained_layout=True)
              con = ConnectionPatch(..., axesA=ax[0], axesB=ax[1])
              fig.add_artist(con)

        """
        if coordsB is None:
            coordsB = coordsA
        self.xy1 = xyA
        self.xy2 = xyB
        self.coords1 = coordsA
        self.coords2 = coordsB
        self.axesA = axesA
        self.axesB = axesB
        super().__init__(posA=(0, 0), posB=(1, 1), arrowstyle=arrowstyle, connectionstyle=connectionstyle, patchA=patchA, patchB=patchB, shrinkA=shrinkA, shrinkB=shrinkB, mutation_scale=mutation_scale, mutation_aspect=mutation_aspect, clip_on=clip_on, **kwargs)
        self._annotation_clip = None

    def _get_xy(self, xy, s, axes=None):
        """Calculate the pixel position of given point."""
        s0 = s
        if axes is None:
            axes = self.axes
        xy = np.array(xy)
        if s in ['figure points', 'axes points']:
            xy *= self.figure.dpi / 72
            s = s.replace('points', 'pixels')
        elif s == 'figure fraction':
            s = self.figure.transFigure
        elif s == 'subfigure fraction':
            s = self.figure.transSubfigure
        elif s == 'axes fraction':
            s = axes.transAxes
        x, y = xy
        if s == 'data':
            trans = axes.transData
            x = float(self.convert_xunits(x))
            y = float(self.convert_yunits(y))
            return trans.transform((x, y))
        elif s == 'offset points':
            if self.xycoords == 'offset points':
                return self._get_xy(self.xy, 'data')
            return self._get_xy(self.xy, self.xycoords) + xy * self.figure.dpi / 72
        elif s == 'polar':
            theta, r = (x, y)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            trans = axes.transData
            return trans.transform((x, y))
        elif s == 'figure pixels':
            bb = self.figure.figbbox
            x = bb.x0 + x if x >= 0 else bb.x1 + x
            y = bb.y0 + y if y >= 0 else bb.y1 + y
            return (x, y)
        elif s == 'subfigure pixels':
            bb = self.figure.bbox
            x = bb.x0 + x if x >= 0 else bb.x1 + x
            y = bb.y0 + y if y >= 0 else bb.y1 + y
            return (x, y)
        elif s == 'axes pixels':
            bb = axes.bbox
            x = bb.x0 + x if x >= 0 else bb.x1 + x
            y = bb.y0 + y if y >= 0 else bb.y1 + y
            return (x, y)
        elif isinstance(s, transforms.Transform):
            return s.transform(xy)
        else:
            raise ValueError(f'{s0} is not a valid coordinate transformation')

    def set_annotation_clip(self, b):
        """
        Set the annotation's clipping behavior.

        Parameters
        ----------
        b : bool or None
            - True: The annotation will be clipped when ``self.xy`` is
              outside the axes.
            - False: The annotation will always be drawn.
            - None: The annotation will be clipped when ``self.xy`` is
              outside the axes and ``self.xycoords == "data"``.
        """
        self._annotation_clip = b
        self.stale = True

    def get_annotation_clip(self):
        """
        Return the clipping behavior.

        See `.set_annotation_clip` for the meaning of the return value.
        """
        return self._annotation_clip

    def _get_path_in_displaycoord(self):
        """Return the mutated path of the arrow in display coordinates."""
        dpi_cor = self._dpi_cor
        posA = self._get_xy(self.xy1, self.coords1, self.axesA)
        posB = self._get_xy(self.xy2, self.coords2, self.axesB)
        path = self.get_connectionstyle()(posA, posB, patchA=self.patchA, patchB=self.patchB, shrinkA=self.shrinkA * dpi_cor, shrinkB=self.shrinkB * dpi_cor)
        path, fillable = self.get_arrowstyle()(path, self.get_mutation_scale() * dpi_cor, self.get_linewidth() * dpi_cor, self.get_mutation_aspect())
        return (path, fillable)

    def _check_xy(self, renderer):
        """Check whether the annotation needs to be drawn."""
        b = self.get_annotation_clip()
        if b or (b is None and self.coords1 == 'data'):
            xy_pixel = self._get_xy(self.xy1, self.coords1, self.axesA)
            if self.axesA is None:
                axes = self.axes
            else:
                axes = self.axesA
            if not axes.contains_point(xy_pixel):
                return False
        if b or (b is None and self.coords2 == 'data'):
            xy_pixel = self._get_xy(self.xy2, self.coords2, self.axesB)
            if self.axesB is None:
                axes = self.axes
            else:
                axes = self.axesB
            if not axes.contains_point(xy_pixel):
                return False
        return True

    def draw(self, renderer):
        if renderer is not None:
            self._renderer = renderer
        if not self.get_visible() or not self._check_xy(renderer):
            return
        super().draw(renderer)
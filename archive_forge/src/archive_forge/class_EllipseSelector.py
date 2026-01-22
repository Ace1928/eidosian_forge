from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
@_docstring.Substitution(_RECTANGLESELECTOR_PARAMETERS_DOCSTRING.replace('__ARTIST_NAME__', 'ellipse'))
class EllipseSelector(RectangleSelector):
    """
    Select an elliptical region of an Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    %s

    Examples
    --------
    :doc:`/gallery/widgets/rectangle_selector`
    """

    def _init_shape(self, **props):
        return Ellipse((0, 0), 0, 1, visible=False, **props)

    def _draw_shape(self, extents):
        x0, x1, y0, y1 = extents
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        center = [x0 + (x1 - x0) / 2.0, y0 + (y1 - y0) / 2.0]
        a = (xmax - xmin) / 2.0
        b = (ymax - ymin) / 2.0
        self._selection_artist.center = center
        self._selection_artist.width = 2 * a
        self._selection_artist.height = 2 * b
        self._selection_artist.angle = self.rotation

    @property
    def _rect_bbox(self):
        x, y = self._selection_artist.center
        width = self._selection_artist.width
        height = self._selection_artist.height
        return (x - width / 2.0, y - height / 2.0, width, height)
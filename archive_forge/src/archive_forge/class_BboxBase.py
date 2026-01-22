import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
class BboxBase(TransformNode):
    """
    The base class of all bounding boxes.

    This class is immutable; `Bbox` is a mutable subclass.

    The canonical representation is as two points, with no
    restrictions on their ordering.  Convenience properties are
    provided to get the left, bottom, right and top edges and width
    and height, but these are not stored explicitly.
    """
    is_bbox = True
    is_affine = True
    if DEBUG:

        @staticmethod
        def _check(points):
            if isinstance(points, np.ma.MaskedArray):
                _api.warn_external('Bbox bounds are a masked array.')
            points = np.asarray(points)
            if any(points[1, :] - points[0, :] == 0):
                _api.warn_external('Singular Bbox.')

    def frozen(self):
        return Bbox(self.get_points().copy())
    frozen.__doc__ = TransformNode.__doc__

    def __array__(self, *args, **kwargs):
        return self.get_points()

    @property
    def x0(self):
        """
        The first of the pair of *x* coordinates that define the bounding box.

        This is not guaranteed to be less than :attr:`x1` (for that, use
        :attr:`xmin`).
        """
        return self.get_points()[0, 0]

    @property
    def y0(self):
        """
        The first of the pair of *y* coordinates that define the bounding box.

        This is not guaranteed to be less than :attr:`y1` (for that, use
        :attr:`ymin`).
        """
        return self.get_points()[0, 1]

    @property
    def x1(self):
        """
        The second of the pair of *x* coordinates that define the bounding box.

        This is not guaranteed to be greater than :attr:`x0` (for that, use
        :attr:`xmax`).
        """
        return self.get_points()[1, 0]

    @property
    def y1(self):
        """
        The second of the pair of *y* coordinates that define the bounding box.

        This is not guaranteed to be greater than :attr:`y0` (for that, use
        :attr:`ymax`).
        """
        return self.get_points()[1, 1]

    @property
    def p0(self):
        """
        The first pair of (*x*, *y*) coordinates that define the bounding box.

        This is not guaranteed to be the bottom-left corner (for that, use
        :attr:`min`).
        """
        return self.get_points()[0]

    @property
    def p1(self):
        """
        The second pair of (*x*, *y*) coordinates that define the bounding box.

        This is not guaranteed to be the top-right corner (for that, use
        :attr:`max`).
        """
        return self.get_points()[1]

    @property
    def xmin(self):
        """The left edge of the bounding box."""
        return np.min(self.get_points()[:, 0])

    @property
    def ymin(self):
        """The bottom edge of the bounding box."""
        return np.min(self.get_points()[:, 1])

    @property
    def xmax(self):
        """The right edge of the bounding box."""
        return np.max(self.get_points()[:, 0])

    @property
    def ymax(self):
        """The top edge of the bounding box."""
        return np.max(self.get_points()[:, 1])

    @property
    def min(self):
        """The bottom-left corner of the bounding box."""
        return np.min(self.get_points(), axis=0)

    @property
    def max(self):
        """The top-right corner of the bounding box."""
        return np.max(self.get_points(), axis=0)

    @property
    def intervalx(self):
        """
        The pair of *x* coordinates that define the bounding box.

        This is not guaranteed to be sorted from left to right.
        """
        return self.get_points()[:, 0]

    @property
    def intervaly(self):
        """
        The pair of *y* coordinates that define the bounding box.

        This is not guaranteed to be sorted from bottom to top.
        """
        return self.get_points()[:, 1]

    @property
    def width(self):
        """The (signed) width of the bounding box."""
        points = self.get_points()
        return points[1, 0] - points[0, 0]

    @property
    def height(self):
        """The (signed) height of the bounding box."""
        points = self.get_points()
        return points[1, 1] - points[0, 1]

    @property
    def size(self):
        """The (signed) width and height of the bounding box."""
        points = self.get_points()
        return points[1] - points[0]

    @property
    def bounds(self):
        """Return (:attr:`x0`, :attr:`y0`, :attr:`width`, :attr:`height`)."""
        (x0, y0), (x1, y1) = self.get_points()
        return (x0, y0, x1 - x0, y1 - y0)

    @property
    def extents(self):
        """Return (:attr:`x0`, :attr:`y0`, :attr:`x1`, :attr:`y1`)."""
        return self.get_points().flatten()

    def get_points(self):
        raise NotImplementedError

    def containsx(self, x):
        """
        Return whether *x* is in the closed (:attr:`x0`, :attr:`x1`) interval.
        """
        x0, x1 = self.intervalx
        return x0 <= x <= x1 or x0 >= x >= x1

    def containsy(self, y):
        """
        Return whether *y* is in the closed (:attr:`y0`, :attr:`y1`) interval.
        """
        y0, y1 = self.intervaly
        return y0 <= y <= y1 or y0 >= y >= y1

    def contains(self, x, y):
        """
        Return whether ``(x, y)`` is in the bounding box or on its edge.
        """
        return self.containsx(x) and self.containsy(y)

    def overlaps(self, other):
        """
        Return whether this bounding box overlaps with the other bounding box.

        Parameters
        ----------
        other : `.BboxBase`
        """
        ax1, ay1, ax2, ay2 = self.extents
        bx1, by1, bx2, by2 = other.extents
        if ax2 < ax1:
            ax2, ax1 = (ax1, ax2)
        if ay2 < ay1:
            ay2, ay1 = (ay1, ay2)
        if bx2 < bx1:
            bx2, bx1 = (bx1, bx2)
        if by2 < by1:
            by2, by1 = (by1, by2)
        return ax1 <= bx2 and bx1 <= ax2 and (ay1 <= by2) and (by1 <= ay2)

    def fully_containsx(self, x):
        """
        Return whether *x* is in the open (:attr:`x0`, :attr:`x1`) interval.
        """
        x0, x1 = self.intervalx
        return x0 < x < x1 or x0 > x > x1

    def fully_containsy(self, y):
        """
        Return whether *y* is in the open (:attr:`y0`, :attr:`y1`) interval.
        """
        y0, y1 = self.intervaly
        return y0 < y < y1 or y0 > y > y1

    def fully_contains(self, x, y):
        """
        Return whether ``x, y`` is in the bounding box, but not on its edge.
        """
        return self.fully_containsx(x) and self.fully_containsy(y)

    def fully_overlaps(self, other):
        """
        Return whether this bounding box overlaps with the other bounding box,
        not including the edges.

        Parameters
        ----------
        other : `.BboxBase`
        """
        ax1, ay1, ax2, ay2 = self.extents
        bx1, by1, bx2, by2 = other.extents
        if ax2 < ax1:
            ax2, ax1 = (ax1, ax2)
        if ay2 < ay1:
            ay2, ay1 = (ay1, ay2)
        if bx2 < bx1:
            bx2, bx1 = (bx1, bx2)
        if by2 < by1:
            by2, by1 = (by1, by2)
        return ax1 < bx2 and bx1 < ax2 and (ay1 < by2) and (by1 < ay2)

    def transformed(self, transform):
        """
        Construct a `Bbox` by statically transforming this one by *transform*.
        """
        pts = self.get_points()
        ll, ul, lr = transform.transform(np.array([pts[0], [pts[0, 0], pts[1, 1]], [pts[1, 0], pts[0, 1]]]))
        return Bbox([ll, [lr[0], ul[1]]])
    coefs = {'C': (0.5, 0.5), 'SW': (0, 0), 'S': (0.5, 0), 'SE': (1.0, 0), 'E': (1.0, 0.5), 'NE': (1.0, 1.0), 'N': (0.5, 1.0), 'NW': (0, 1.0), 'W': (0, 0.5)}

    def anchored(self, c, container=None):
        """
        Return a copy of the `Bbox` anchored to *c* within *container*.

        Parameters
        ----------
        c : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
            Either an (*x*, *y*) pair of relative coordinates (0 is left or
            bottom, 1 is right or top), 'C' (center), or a cardinal direction
            ('SW', southwest, is bottom left, etc.).
        container : `Bbox`, optional
            The box within which the `Bbox` is positioned.

        See Also
        --------
        .Axes.set_anchor
        """
        if container is None:
            _api.warn_deprecated('3.8', message='Calling anchored() with no container bbox returns a frozen copy of the original bbox and is deprecated since %(since)s.')
            container = self
        l, b, w, h = container.bounds
        L, B, W, H = self.bounds
        cx, cy = self.coefs[c] if isinstance(c, str) else c
        return Bbox(self._points + [l + cx * (w - W) - L, b + cy * (h - H) - B])

    def shrunk(self, mx, my):
        """
        Return a copy of the `Bbox`, shrunk by the factor *mx*
        in the *x* direction and the factor *my* in the *y* direction.
        The lower left corner of the box remains unchanged.  Normally
        *mx* and *my* will be less than 1, but this is not enforced.
        """
        w, h = self.size
        return Bbox([self._points[0], self._points[0] + [mx * w, my * h]])

    def shrunk_to_aspect(self, box_aspect, container=None, fig_aspect=1.0):
        """
        Return a copy of the `Bbox`, shrunk so that it is as
        large as it can be while having the desired aspect ratio,
        *box_aspect*.  If the box coordinates are relative (i.e.
        fractions of a larger box such as a figure) then the
        physical aspect ratio of that figure is specified with
        *fig_aspect*, so that *box_aspect* can also be given as a
        ratio of the absolute dimensions, not the relative dimensions.
        """
        if box_aspect <= 0 or fig_aspect <= 0:
            raise ValueError("'box_aspect' and 'fig_aspect' must be positive")
        if container is None:
            container = self
        w, h = container.size
        H = w * box_aspect / fig_aspect
        if H <= h:
            W = w
        else:
            W = h * fig_aspect / box_aspect
            H = h
        return Bbox([self._points[0], self._points[0] + (W, H)])

    def splitx(self, *args):
        """
        Return a list of new `Bbox` objects formed by splitting the original
        one with vertical lines at fractional positions given by *args*.
        """
        xf = [0, *args, 1]
        x0, y0, x1, y1 = self.extents
        w = x1 - x0
        return [Bbox([[x0 + xf0 * w, y0], [x0 + xf1 * w, y1]]) for xf0, xf1 in zip(xf[:-1], xf[1:])]

    def splity(self, *args):
        """
        Return a list of new `Bbox` objects formed by splitting the original
        one with horizontal lines at fractional positions given by *args*.
        """
        yf = [0, *args, 1]
        x0, y0, x1, y1 = self.extents
        h = y1 - y0
        return [Bbox([[x0, y0 + yf0 * h], [x1, y0 + yf1 * h]]) for yf0, yf1 in zip(yf[:-1], yf[1:])]

    def count_contains(self, vertices):
        """
        Count the number of vertices contained in the `Bbox`.
        Any vertices with a non-finite x or y value are ignored.

        Parameters
        ----------
        vertices : (N, 2) array
        """
        if len(vertices) == 0:
            return 0
        vertices = np.asarray(vertices)
        with np.errstate(invalid='ignore'):
            return ((self.min < vertices) & (vertices < self.max)).all(axis=1).sum()

    def count_overlaps(self, bboxes):
        """
        Count the number of bounding boxes that overlap this one.

        Parameters
        ----------
        bboxes : sequence of `.BboxBase`
        """
        return count_bboxes_overlapping_bbox(self, np.atleast_3d([np.array(x) for x in bboxes]))

    def expanded(self, sw, sh):
        """
        Construct a `Bbox` by expanding this one around its center by the
        factors *sw* and *sh*.
        """
        width = self.width
        height = self.height
        deltaw = (sw * width - width) / 2.0
        deltah = (sh * height - height) / 2.0
        a = np.array([[-deltaw, -deltah], [deltaw, deltah]])
        return Bbox(self._points + a)

    @_api.rename_parameter('3.8', 'p', 'w_pad')
    def padded(self, w_pad, h_pad=None):
        """
        Construct a `Bbox` by padding this one on all four sides.

        Parameters
        ----------
        w_pad : float
            Width pad
        h_pad : float, optional
            Height pad.  Defaults to *w_pad*.

        """
        points = self.get_points()
        if h_pad is None:
            h_pad = w_pad
        return Bbox(points + [[-w_pad, -h_pad], [w_pad, h_pad]])

    def translated(self, tx, ty):
        """Construct a `Bbox` by translating this one by *tx* and *ty*."""
        return Bbox(self._points + (tx, ty))

    def corners(self):
        """
        Return the corners of this rectangle as an array of points.

        Specifically, this returns the array
        ``[[x0, y0], [x0, y1], [x1, y0], [x1, y1]]``.
        """
        (x0, y0), (x1, y1) = self.get_points()
        return np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])

    def rotated(self, radians):
        """
        Return the axes-aligned bounding box that bounds the result of rotating
        this `Bbox` by an angle of *radians*.
        """
        corners = self.corners()
        corners_rotated = Affine2D().rotate(radians).transform(corners)
        bbox = Bbox.unit()
        bbox.update_from_data_xy(corners_rotated, ignore=True)
        return bbox

    @staticmethod
    def union(bboxes):
        """Return a `Bbox` that contains all of the given *bboxes*."""
        if not len(bboxes):
            raise ValueError("'bboxes' cannot be empty")
        x0 = np.min([bbox.xmin for bbox in bboxes])
        x1 = np.max([bbox.xmax for bbox in bboxes])
        y0 = np.min([bbox.ymin for bbox in bboxes])
        y1 = np.max([bbox.ymax for bbox in bboxes])
        return Bbox([[x0, y0], [x1, y1]])

    @staticmethod
    def intersection(bbox1, bbox2):
        """
        Return the intersection of *bbox1* and *bbox2* if they intersect, or
        None if they don't.
        """
        x0 = np.maximum(bbox1.xmin, bbox2.xmin)
        x1 = np.minimum(bbox1.xmax, bbox2.xmax)
        y0 = np.maximum(bbox1.ymin, bbox2.ymin)
        y1 = np.minimum(bbox1.ymax, bbox2.ymax)
        return Bbox([[x0, y0], [x1, y1]]) if x0 <= x1 and y0 <= y1 else None
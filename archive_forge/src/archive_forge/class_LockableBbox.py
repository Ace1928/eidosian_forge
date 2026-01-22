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
class LockableBbox(BboxBase):
    """
    A `Bbox` where some elements may be locked at certain values.

    When the child bounding box changes, the bounds of this bbox will update
    accordingly with the exception of the locked elements.
    """

    def __init__(self, bbox, x0=None, y0=None, x1=None, y1=None, **kwargs):
        """
        Parameters
        ----------
        bbox : `Bbox`
            The child bounding box to wrap.

        x0 : float or None
            The locked value for x0, or None to leave unlocked.

        y0 : float or None
            The locked value for y0, or None to leave unlocked.

        x1 : float or None
            The locked value for x1, or None to leave unlocked.

        y1 : float or None
            The locked value for y1, or None to leave unlocked.

        """
        if not bbox.is_bbox:
            raise ValueError("'bbox' is not a bbox")
        super().__init__(**kwargs)
        self._bbox = bbox
        self.set_children(bbox)
        self._points = None
        fp = [x0, y0, x1, y1]
        mask = [val is None for val in fp]
        self._locked_points = np.ma.array(fp, float, mask=mask).reshape((2, 2))
    __str__ = _make_str_method('_bbox', '_locked_points')

    def get_points(self):
        if self._invalid:
            points = self._bbox.get_points()
            self._points = np.where(self._locked_points.mask, points, self._locked_points)
            self._invalid = 0
        return self._points
    if DEBUG:
        _get_points = get_points

        def get_points(self):
            points = self._get_points()
            self._check(points)
            return points

    @property
    def locked_x0(self):
        """
        float or None: The value used for the locked x0.
        """
        if self._locked_points.mask[0, 0]:
            return None
        else:
            return self._locked_points[0, 0]

    @locked_x0.setter
    def locked_x0(self, x0):
        self._locked_points.mask[0, 0] = x0 is None
        self._locked_points.data[0, 0] = x0
        self.invalidate()

    @property
    def locked_y0(self):
        """
        float or None: The value used for the locked y0.
        """
        if self._locked_points.mask[0, 1]:
            return None
        else:
            return self._locked_points[0, 1]

    @locked_y0.setter
    def locked_y0(self, y0):
        self._locked_points.mask[0, 1] = y0 is None
        self._locked_points.data[0, 1] = y0
        self.invalidate()

    @property
    def locked_x1(self):
        """
        float or None: The value used for the locked x1.
        """
        if self._locked_points.mask[1, 0]:
            return None
        else:
            return self._locked_points[1, 0]

    @locked_x1.setter
    def locked_x1(self, x1):
        self._locked_points.mask[1, 0] = x1 is None
        self._locked_points.data[1, 0] = x1
        self.invalidate()

    @property
    def locked_y1(self):
        """
        float or None: The value used for the locked y1.
        """
        if self._locked_points.mask[1, 1]:
            return None
        else:
            return self._locked_points[1, 1]

    @locked_y1.setter
    def locked_y1(self, y1):
        self._locked_points.mask[1, 1] = y1 is None
        self._locked_points.data[1, 1] = y1
        self.invalidate()
from collections import namedtuple
import contextlib
from functools import cache, wraps
import inspect
from inspect import Signature, Parameter
import logging
from numbers import Number, Real
import re
import warnings
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .colors import BoundaryNorm
from .cm import ScalarMappable
from .path import Path
from .transforms import (BboxBase, Bbox, IdentityTransform, Transform, TransformedBbox,
def _fully_clipped_to_axes(self):
    """
        Return a boolean flag, ``True`` if the artist is clipped to the Axes
        and can thus be skipped in layout calculations. Requires `get_clip_on`
        is True, one of `clip_box` or `clip_path` is set, ``clip_box.extents``
        is equivalent to ``ax.bbox.extents`` (if set), and ``clip_path._patch``
        is equivalent to ``ax.patch`` (if set).
        """
    clip_box = self.get_clip_box()
    clip_path = self.get_clip_path()
    return self.axes is not None and self.get_clip_on() and (clip_box is not None or clip_path is not None) and (clip_box is None or np.all(clip_box.extents == self.axes.bbox.extents)) and (clip_path is None or (isinstance(clip_path, TransformedPatchPath) and clip_path._patch is self.axes.patch))
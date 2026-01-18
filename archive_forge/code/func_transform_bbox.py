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
def transform_bbox(self, bbox):
    """
        Transform the given bounding box.

        For smarter transforms including caching (a common requirement in
        Matplotlib), see `TransformedBbox`.
        """
    return Bbox(self.transform(bbox.get_points()))
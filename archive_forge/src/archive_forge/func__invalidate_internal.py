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
def _invalidate_internal(self, level, invalidating_node):
    if invalidating_node is self._a and (not self._b.is_affine):
        level = Transform._INVALID_FULL
    super()._invalidate_internal(level, invalidating_node)
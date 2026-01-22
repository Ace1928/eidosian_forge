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
def _revalidate(self):
    patch_path = self._patch.get_path()
    if patch_path != self._path:
        self._path = patch_path
        self._transformed_path = None
    super()._revalidate()
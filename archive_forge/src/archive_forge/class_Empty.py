from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
class Empty:
    """
        Proxy object to represent empty/null dataspaces (a.k.a H5S_NULL).

        This can have an associated dtype, but has no shape or data. This is not
        the same as an array with shape (0,).
    """
    shape = None
    size = None

    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)

    def __eq__(self, other):
        if isinstance(other, Empty) and self.dtype == other.dtype:
            return True
        return False

    def __repr__(self):
        return 'Empty(dtype={0!r})'.format(self.dtype)
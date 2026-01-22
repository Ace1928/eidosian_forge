import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
@staticmethod
def _boundsspec2slicespec(boundsspec, scs):
    """
        Convert an iterable boundsspec (supplying l,b,r,t of a
        BoundingRegion) into a Slice specification.

        Includes all units whose centers are within the specified
        sheet-coordinate bounds specified by boundsspec.

        Exact inverse of _slicespec2boundsspec().
        """
    l, b, r, t = boundsspec
    t_m, l_m = scs.sheet2matrix(l, t)
    b_m, r_m = scs.sheet2matrix(r, b)
    l_idx = int(np.ceil(l_m - 0.5))
    t_idx = int(np.ceil(t_m - 0.5))
    r_idx = int(np.floor(r_m + 0.5))
    b_idx = int(np.floor(b_m + 0.5))
    return (t_idx, b_idx, l_idx, r_idx)
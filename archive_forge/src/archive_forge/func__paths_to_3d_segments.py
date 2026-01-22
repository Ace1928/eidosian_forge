import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def _paths_to_3d_segments(paths, zs=0, zdir='z'):
    """Convert paths from a collection object to 3D segments."""
    if not np.iterable(zs):
        zs = np.broadcast_to(zs, len(paths))
    elif len(zs) != len(paths):
        raise ValueError('Number of z-coordinates does not match paths.')
    segs = [_path_to_3d_segment(path, pathz, zdir) for path, pathz in zip(paths, zs)]
    return segs
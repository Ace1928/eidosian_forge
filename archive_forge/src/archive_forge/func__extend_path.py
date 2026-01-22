import math
import numpy as np
import matplotlib as mpl
from matplotlib.patches import _Style, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform
def _extend_path(self, path, mutation_size=10):
    """
            Extend the path to make a room for drawing arrow.
            """
    (x0, y0), (x1, y1) = path.vertices[-2:]
    theta = math.atan2(y1 - y0, x1 - x0)
    x2 = x1 + math.cos(theta) * mutation_size
    y2 = y1 + math.sin(theta) * mutation_size
    if path.codes is None:
        return Path(np.concatenate([path.vertices, [[x2, y2]]]))
    else:
        return Path(np.concatenate([path.vertices, [[x2, y2]]]), np.concatenate([path.codes, [Path.LINETO]]))
import math
import numpy as np
from numpy import ma
from matplotlib import _api, cbook, _docstring
import matplotlib.artist as martist
import matplotlib.collections as mcollections
from matplotlib.patches import CirclePolygon
import matplotlib.text as mtext
import matplotlib.transforms as transforms
def _angles_lengths(self, XY, U, V, eps=1):
    xy = self.axes.transData.transform(XY)
    uv = np.column_stack((U, V))
    xyp = self.axes.transData.transform(XY + eps * uv)
    dxy = xyp - xy
    angles = np.arctan2(dxy[:, 1], dxy[:, 0])
    lengths = np.hypot(*dxy.T) / eps
    return (angles, lengths)
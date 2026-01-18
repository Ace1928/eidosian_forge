import numpy as np
from matplotlib import _api
def world_transformation(xmin, xmax, ymin, ymax, zmin, zmax, pb_aspect=None):
    """
    Produce a matrix that scales homogeneous coords in the specified ranges
    to [0, 1], or [0, pb_aspect[i]] if the plotbox aspect ratio is specified.
    """
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    if pb_aspect is not None:
        ax, ay, az = pb_aspect
        dx /= ax
        dy /= ay
        dz /= az
    return np.array([[1 / dx, 0, 0, -xmin / dx], [0, 1 / dy, 0, -ymin / dy], [0, 0, 1 / dz, -zmin / dz], [0, 0, 0, 1]])
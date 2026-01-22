import functools
from itertools import chain
import numpy as np
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, IdentityTransform
from .axislines import (
from .axis_artist import AxisArtist
from .grid_finder import GridFinder
def _value_and_jacobian(func, xs, ys, xlims, ylims):
    """
    Compute *func* and its derivatives along x and y at positions *xs*, *ys*,
    while ensuring that finite difference calculations don't try to evaluate
    values outside of *xlims*, *ylims*.
    """
    eps = np.finfo(float).eps ** (1 / 2)
    val = func(xs, ys)
    xlo, xhi = sorted(xlims)
    dxlo = xs - xlo
    dxhi = xhi - xs
    xeps = np.take([-1, 1], dxhi >= dxlo) * np.minimum(eps, np.maximum(dxlo, dxhi))
    val_dx = func(xs + xeps, ys)
    ylo, yhi = sorted(ylims)
    dylo = ys - ylo
    dyhi = yhi - ys
    yeps = np.take([-1, 1], dyhi >= dylo) * np.minimum(eps, np.maximum(dylo, dyhi))
    val_dy = func(xs, ys + yeps)
    return (val, (val_dx - val) / xeps, (val_dy - val) / yeps)
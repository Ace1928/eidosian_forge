import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator
def _find_out_of_bounds(self, xi):
    out_of_bounds = cp.zeros(xi.shape[1], dtype=bool)
    for x, grid in zip(xi, self.grid):
        out_of_bounds += x < grid[0]
        out_of_bounds += x > grid[-1]
    return out_of_bounds
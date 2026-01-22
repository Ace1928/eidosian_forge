import warnings
import numpy as np
from scipy import interpolate, stats
class _Grid:
    """Create Grid values and indices, grid in [0, 1]^d

    This class creates a regular grid in a d dimensional hyper cube.

    Intended for internal use, implementation might change without warning.


    Parameters
    ----------
    k_grid : tuple or array_like
        number of elements for axes, this defines k_grid - 1 equal sized
        intervals of [0, 1] for each axis.
    eps : float
        If eps is not zero, then x values will be clipped to [eps, 1 - eps],
        i.e. to the interior of the unit interval or hyper cube.


    Attributes
    ----------
    k_grid : list of number of grid points
    x_marginal: list of 1-dimensional marginal values
    idx_flat: integer array with indices
    x_flat: flattened grid values,
        rows are grid points, columns represent variables or axis.
        ``x_flat`` is currently also 2-dim in the univariate 1-dim grid case.

    """

    def __init__(self, k_grid, eps=0):
        self.k_grid = k_grid
        x_marginal = [np.arange(ki) / (ki - 1) for ki in k_grid]
        idx_flat = np.column_stack(np.unravel_index(np.arange(np.prod(k_grid)), k_grid)).astype(float)
        x_flat = idx_flat / idx_flat.max(0)
        if eps != 0:
            x_marginal = [np.clip(xi, eps, 1 - eps) for xi in x_marginal]
            x_flat = np.clip(x_flat, eps, 1 - eps)
        self.x_marginal = x_marginal
        self.idx_flat = idx_flat
        self.x_flat = x_flat
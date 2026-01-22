import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from ..morphology._util import _raveled_offsets_and_distances
from ..util._map_array import map_array
def _weighted_abs_diff(values0, values1, distances):
    """A default edge function for complete image graphs.

    A pixel graph on an image with no edge values and no mask is a very
    boring regular lattice, so we define a default edge weight to be the
    absolute difference between values *weighted* by the distance
    between them.

    Parameters
    ----------
    values0 : array
        The pixel values for each node.
    values1 : array
        The pixel values for each neighbor.
    distances : array
        The distance between each node and its neighbor.

    Returns
    -------
    edge_values : array of float
        The computed values: abs(values0 - values1) * distances.
    """
    return np.abs(values0 - values1) * distances
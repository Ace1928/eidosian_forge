import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from ..morphology._util import _raveled_offsets_and_distances
from ..util._map_array import map_array
Find the pixel with the highest closeness centrality.

    Closeness centrality is the inverse of the total sum of shortest distances
    from a node to every other node.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        The sparse matrix representation of the graph.
    nodes : array of int
        The raveled index of each node in graph in the image. If not provided,
        the returned value will be the index in the input graph.
    shape : tuple of int
        The shape of the image in which the nodes are embedded. If provided,
        the returned coordinates are a NumPy multi-index of the same
        dimensionality as the input shape. Otherwise, the returned coordinate
        is the raveled index provided in `nodes`.
    partition_size : int
        This function computes the shortest path distance between every pair
        of nodes in the graph. This can result in a very large (N*N) matrix.
        As a simple performance tweak, the distance values are computed in
        lots of `partition_size`, resulting in a memory requirement of only
        partition_size*N.

    Returns
    -------
    position : int or tuple of int
        If shape is given, the coordinate of the central pixel in the image.
        Otherwise, the raveled index of that pixel.
    distances : array of float
        The total sum of distances from each node to each other reachable
        node.
    
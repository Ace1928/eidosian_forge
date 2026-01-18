import numpy as np
from scipy import sparse, stats
from scipy.sparse import linalg
from pygsp import graphs, filters, utils
Compute a multiresolution of trees

    Parameters
    ----------
    G : Graph
        Graph structure of a tree.
    Nlevel : Number of times to downsample and coarsen the tree
    root : int
        The index of the root of the tree. (default = 1)
    reduction_method : str
        The graph reduction method (default = 'resistance_distance')
    compute_full_eigen : bool
        To also compute the graph Laplacian eigenvalues for every tree in the sequence

    Returns
    -------
    Gs : ndarray
        Ndarray, with each element containing a graph structure represent a reduced tree.
    subsampled_vertex_indices : ndarray
        Indices of the vertices of the previous tree that are kept for the subsequent tree.

    
import traceback
import numpy as np
from scipy import sparse, spatial
from pygsp import utils
from pygsp.graphs import Graph  # prevent circular import in Python < 3.5
Nearest-neighbor graph from given point cloud.

    Parameters
    ----------
    Xin : ndarray
        Input points, Should be an `N`-by-`d` matrix, where `N` is the number
        of nodes in the graph and `d` is the dimension of the feature space.
    NNtype : string, optional
        Type of nearest neighbor graph to create. The options are 'knn' for
        k-Nearest Neighbors or 'radius' for epsilon-Nearest Neighbors (default
        is 'knn').
    use_flann : bool, optional
        Use Fast Library for Approximate Nearest Neighbors (FLANN) or not.
        (default is False)
    center : bool, optional
        Center the data so that it has zero mean (default is True)
    rescale : bool, optional
        Rescale the data so that it lies in a l2-sphere (default is True)
    k : int, optional
        Number of neighbors for knn (default is 10)
    sigma : float, optional
        Width parameter of the similarity kernel (default is 0.1)
    epsilon : float, optional
        Radius for the epsilon-neighborhood search (default is 0.01)
    gtype : string, optional
        The type of graph (default is 'nearest neighbors')
    plotting : dict, optional
        Dictionary of plotting parameters. See :obj:`pygsp.plotting`.
        (default is {})
    symmetrize_type : string, optional
        Type of symmetrization to use for the adjacency matrix. See
        :func:`pygsp.utils.symmetrization` for the options.
        (default is 'average')
    dist_type : string, optional
        Type of distance to compute. See
        :func:`pyflann.index.set_distance_type` for possible options.
        (default is 'euclidean')
    order : float, optional
        Only used if dist_type is 'minkowski'; represents the order of the
        Minkowski distance. (default is 0)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> X = np.random.RandomState(42).uniform(size=(30, 2))
    >>> G = graphs.NNGraph(X)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=5)
    >>> G.plot(ax=axes[1])

    
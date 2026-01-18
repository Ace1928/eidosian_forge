from .. import utils
from . import r_function
import numpy as np
import pandas as pd
import warnings
Perform lineage inference with Slingshot.

    Given a reduced-dimensional data matrix n by p and a vector of cluster labels
    (or matrix of soft cluster assignments, potentially including a -1 label for
    "unclustered"), this function performs lineage inference using a cluster-based
    minimum spanning tree and constructing simulatenous principal curves for branching
    paths through the tree.

    For more details, read about Slingshot on GitHub_ and Bioconductor_.

    .. _GitHub: https://github.com/kstreet13/slingshot
    .. _Bioconductor: https://bioconductor.org/packages/release/bioc/html/slingshot.html

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_dimensions]
        matrix of (reduced dimension) coordinates
        to be used for lineage inference.
    cluster_labels : list-like, shape=[n_samples]
        a vector of cluster labels, optionally including -1's for "unclustered."
    start_cluster : string, optional (default: None)
        indicates the cluster(s) of origin.
        Lineages will be represented by paths coming out of this cluster.
    end_cluster : string, optional (default: None)
        indicates the cluster(s) which will be forced leaf nodes.
        This introduces a constraint on the MST algorithm.
    distance : callable, optional (default: None)
        method for calculating distances between clusters.
        Must take two matrices as input, corresponding to subsets of reduced_dim.
        If the minimum cluster size is larger than the number dimensions,
        the default is to use the joint covariance matrix to find squared distance
        between cluster centers. If not, the default is to use the diagonal of the
        joint covariance matrix. Not currently implemented
    omega : float, optional (default: None)
        this granularity parameter determines the distance between every
        real cluster and the artificial cluster.
        It is parameterized such that this distance is omega / 2,
        making omega the maximum distance between two connected clusters.
        By default, omega = Inf.
    shrink : boolean or float, optional (default: True)
        boolean or numeric between 0 and 1, determines whether and how much to shrink
        branching lineages toward their average prior to the split.
    extend : {'y', 'n', 'pc1'}, optional (default: "y")
        how to handle root and leaf clusters of lineages when
        constructing the initial, piece-wise linear curve.
    reweight : boolean, optional (default: True)
        whether to allow cells shared between lineages to be reweighted during
        curve-fitting. If True, cells shared between lineages will be iteratively
        reweighted based on the quantiles of their projection distances to each curve.
    reassign : boolean, optional (default: True)
        whether to reassign cells to lineages at each iteration.
        If True, cells will be added to a lineage when their
        projection distance to the curve is less than the median
        distance for all cells currently assigned to the lineage.
        Additionally, shared cells will be removed from a lineage if
        their projection distance to the curve is above the 90th
        percentile and their weight along the curve is less than 0.1.
    thresh : float, optional (default: 0.001)
        determines the convergence criterion. Percent change in the
        total distance from cells to their projections along curves
        must be less than thresh.
    max_iter : int, optional (default: 15)
        maximum number of iterations
    stretch : int, optional (default: 2)
        factor between 0 and 2 by which curves can be extrapolated beyond endpoints
    smoother : {"smooth.spline", "lowess", "periodic_lowess"},
        optional (default: "smooth.spline")
        choice of smoother. "periodic_lowess" allows one to fit closed
        curves. Beware, you may want to use iter = 0 with "lowess".
    shrink_method : string, optional (default: "cosine")
        how to determine the appropriate amount of shrinkage for a
        branching lineage. Accepted values: "gaussian", "rectangular",
        "triangular", "epanechnikov", "biweight", "triweight",
        "cosine", "optcosine", "density".
    allow_breaks : boolean, optional (default: True)
        determines whether curves that branch very close to the origin
        should be allowed to have different starting points.
    seed : int or None, optional (default: None)
        Seed to use for generating random numbers.
    verbose : int, optional (default: 1)
        Logging verbosity between 0 and 2.

    Returns
    -------
    slingshot : dict
        Contains the following keys:
    pseudotime : array-like, shape=[n_samples, n_curves]
        Pseudotime projection of each cell onto each principal curve.
        Value is `np.nan` if the cell does not lie on the curve
    branch : list-like, shape=[n_samples]
        Branch assignment for each cell
    curves : array_like, shape=[n_curves, n_samples, n_dimensions]
        Coordinates of each principle curve in the reduced dimension

    Examples
    --------
    >>> import scprep
    >>> import phate
    >>> data, clusters = phate.tree.gen_dla(n_branch=4, n_dim=200, branch_length=200)
    >>> phate_op = phate.PHATE()
    >>> data_phate = phate_op.fit_transform(data)
    >>> slingshot = scprep.run.Slingshot(data_phate, clusters)
    >>> ax = scprep.plot.scatter2d(
    ...     data_phate,
    ...     c=slingshot['pseudotime'][:,0],
    ...     cmap='magma',
    ...     legend_title='Branch 1'
    ... )
    >>> scprep.plot.scatter2d(
    ...     data_phate,
    ...     c=slingshot['pseudotime'][:,1],
    ...     cmap='viridis',
    ...     ax=ax,
    ...     ticks=False,
    ...     label_prefix='PHATE',
    ...     legend_title='Branch 2'
    ...     )
    >>> for curve in slingshot['curves']:
    ...     ax.plot(curve[:,0], curve[:,1], c='black')
    >>> ax = scprep.plot.scatter2d(data_phate, c=slingshot['branch'],
    ...                        legend_title='Branch', ticks=False, label_prefix='PHATE')
    >>> for curve in slingshot['curves']:
    ...     ax.plot(curve[:,0], curve[:,1], c='black')
    
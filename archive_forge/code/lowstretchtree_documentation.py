import numpy as np
from scipy import sparse
from . import Graph  # prevent circular import in Python < 3.5
Low stretch tree.

    Build the root of a low stretch tree on a grid of points. There are
    :math:`2k` points on each side of the grid, and therefore :math:`2^{2k}`
    vertices in total. The edge weights are all equal to 1.

    Parameters
    ----------
    k : int
        :math:`2^k` points on each side of the grid of vertices.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.LowStretchTree(k=2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> G.plot(ax=axes[1])

    
import numpy as np
from scipy import sparse
from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5
Ring graph with randomly sampled nodes.

    Parameters
    ----------
    N : int
        Number of vertices.
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.RandomRing(N=10, seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> G.plot(ax=axes[1])
    >>> _ = axes[1].set_xlim(-1.1, 1.1)
    >>> _ = axes[1].set_ylim(-1.1, 1.1)

    
import numpy as np
from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5
Sensor network.

    Parameters
    ----------
    N : int
        Number of vertices (default = 64). Values of 64 and 500 yield
        pre-computed and saved graphs. Other values yield randomly generated
        graphs.
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.DavidSensorNet()
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> G.plot(ax=axes[1])

    
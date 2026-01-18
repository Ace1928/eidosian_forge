import numpy as np
from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5
Sampled Swiss roll manifold.

    Parameters
    ----------
    N : int
        Number of vertices (default = 400)
    a : int
        (default = 1)
    b : int
        (default = 4)
    dim : int
        (default = 3)
    thresh : float
        (default = 1e-6)
    s : float
        sigma (default =  sqrt(2./N))
    noise : bool
        Wether to add noise or not (default = False)
    srtype : str
        Swiss roll Type, possible arguments are 'uniform' or 'classic'
        (default = 'uniform')
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.SwissRoll(N=200, seed=42)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1)
    >>> G.plot(ax=ax2)

    
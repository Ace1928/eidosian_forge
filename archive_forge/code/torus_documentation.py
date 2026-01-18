import numpy as np
from scipy import sparse
from . import Graph  # prevent circular import in Python < 3.5
Sampled torus manifold.

    Parameters
    ----------
    Nv : int
        Number of vertices along the first dimension (default is 16)
    Mv : int
        Number of vertices along the second dimension (default is Nv)

    References
    ----------
    See :cite:`strang1999discrete` for more informations.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Torus(10)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> G.plot(ax=ax2)
    >>> _ = ax2.set_zlim(-1.5, 1.5)

    
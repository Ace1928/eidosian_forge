from pygsp import utils
from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5
Stanford bunny (NN-graph).

    References
    ----------
    See :cite:`turk1994zippered`.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Bunny()
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=0.1)
    >>> G.plot(ax=ax2)

    
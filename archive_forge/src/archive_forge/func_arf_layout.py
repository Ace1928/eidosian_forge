import networkx as nx
from networkx.utils import np_random_state
def arf_layout(G, pos=None, scaling=1, a=1.1, etol=1e-06, dt=0.001, max_iter=1000):
    """Arf layout for networkx

    The attractive and repulsive forces (arf) layout [1]
    improves the spring layout in three ways. First, it
    prevents congestion of highly connected nodes due to
    strong forcing between nodes. Second, it utilizes the
    layout space more effectively by preventing large gaps
    that spring layout tends to create. Lastly, the arf
    layout represents symmetries in the layout better than
    the default spring layout.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Networkx graph.
    pos : dict
        Initial  position of  the nodes.  If set  to None  a
        random layout will be used.
    scaling : float
        Scales the radius of the circular layout space.
    a : float
        Strength of springs between connected nodes. Should be larger than 1. The greater a, the clearer the separation ofunconnected sub clusters.
    etol : float
        Gradient sum of spring forces must be larger than `etol` before successful termination.
    dt : float
        Time step for force differential equation simulations.
    max_iter : int
        Max iterations before termination of the algorithm.

    References
    .. [1] "Self-Organization Applied to Dynamic Network Layout", M. Geipel,
            International Journal of Modern Physics C, 2007, Vol 18, No 10, pp. 1537-1549.
            https://doi.org/10.1142/S0129183107011558 https://arxiv.org/abs/0704.1748

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.grid_graph((5, 5))
    >>> pos = nx.arf_layout(G)

    """
    import warnings
    import numpy as np
    if a <= 1:
        msg = 'The parameter a should be larger than 1'
        raise ValueError(msg)
    pos_tmp = nx.random_layout(G)
    if pos is None:
        pos = pos_tmp
    else:
        for node in G.nodes():
            if node not in pos:
                pos[node] = pos_tmp[node].copy()
    N = len(G)
    if N == 0:
        return pos
    K = np.ones((N, N)) - np.eye(N)
    node_order = {node: i for i, node in enumerate(G)}
    for x, y in G.edges():
        if x != y:
            idx, jdx = (node_order[i] for i in (x, y))
            K[idx, jdx] = a
    p = np.asarray(list(pos.values()))
    rho = scaling * np.sqrt(N)
    error = etol + 1
    n_iter = 0
    while error > etol:
        diff = p[:, np.newaxis] - p[np.newaxis]
        A = np.linalg.norm(diff, axis=-1)[..., np.newaxis]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            change = K[..., np.newaxis] * diff - rho / A * diff
        change = np.nansum(change, axis=0)
        p += change * dt
        error = np.linalg.norm(change, axis=-1).sum()
        if n_iter > max_iter:
            break
        n_iter += 1
    return dict(zip(G.nodes(), p))
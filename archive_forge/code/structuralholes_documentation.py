import networkx as nx
Returns the local constraint on the node ``u`` with respect to
    the node ``v`` in the graph ``G``.

    Formally, the *local constraint on u with respect to v*, denoted
    $\ell(v)$, is defined by

    .. math::

       \ell(u, v) = \left(p_{uv} + \sum_{w \in N(v)} p_{uw} p_{wv}\right)^2,

    where $N(v)$ is the set of neighbors of $v$ and $p_{uv}$ is the
    normalized mutual weight of the (directed or undirected) edges
    joining $u$ and $v$, for each vertex $u$ and $v$ [1]_. The *mutual
    weight* of $u$ and $v$ is the sum of the weights of edges joining
    them (edge weights are assumed to be one if the graph is
    unweighted).

    Parameters
    ----------
    G : NetworkX graph
        The graph containing ``u`` and ``v``. This can be either
        directed or undirected.

    u : node
        A node in the graph ``G``.

    v : node
        A node in the graph ``G``.

    weight : None or string, optional
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    Returns
    -------
    float
        The constraint of the node ``v`` in the graph ``G``.

    See also
    --------
    constraint

    References
    ----------
    .. [1] Burt, Ronald S.
           "Structural holes and good ideas".
           American Journal of Sociology (110): 349–399.

    
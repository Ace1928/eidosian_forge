import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch(edge_attrs='weight')
def trophic_levels(G, weight='weight'):
    """Compute the trophic levels of nodes.

    The trophic level of a node $i$ is

    .. math::

        s_i = 1 + \\frac{1}{k^{in}_i} \\sum_{j} a_{ij} s_j

    where $k^{in}_i$ is the in-degree of i

    .. math::

        k^{in}_i = \\sum_{j} a_{ij}

    and nodes with $k^{in}_i = 0$ have $s_i = 1$ by convention.

    These are calculated using the method outlined in Levine [1]_.

    Parameters
    ----------
    G : DiGraph
        A directed networkx graph

    Returns
    -------
    nodes : dict
        Dictionary of nodes with trophic level as the value.

    References
    ----------
    .. [1] Stephen Levine (1980) J. theor. Biol. 83, 195-207
    """
    import numpy as np
    a = nx.adjacency_matrix(G, weight=weight).T.toarray()
    rowsum = np.sum(a, axis=1)
    p = a[rowsum != 0][:, rowsum != 0]
    p = p / rowsum[rowsum != 0][:, np.newaxis]
    nn = p.shape[0]
    i = np.eye(nn)
    try:
        n = np.linalg.inv(i - p)
    except np.linalg.LinAlgError as err:
        msg = 'Trophic levels are only defined for graphs where every ' + 'node has a path from a basal node (basal nodes are nodes ' + 'with no incoming edges).'
        raise nx.NetworkXError(msg) from err
    y = n.sum(axis=1) + 1
    levels = {}
    zero_node_ids = (node_id for node_id, degree in G.in_degree if degree == 0)
    for node_id in zero_node_ids:
        levels[node_id] = 1
    nonzero_node_ids = (node_id for node_id, degree in G.in_degree if degree != 0)
    for i, node_id in enumerate(nonzero_node_ids):
        levels[node_id] = y[i]
    return levels
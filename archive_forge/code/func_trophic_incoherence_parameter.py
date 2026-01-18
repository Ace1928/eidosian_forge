import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch(edge_attrs='weight')
def trophic_incoherence_parameter(G, weight='weight', cannibalism=False):
    """Compute the trophic incoherence parameter of a graph.

    Trophic coherence is defined as the homogeneity of the distribution of
    trophic distances: the more similar, the more coherent. This is measured by
    the standard deviation of the trophic differences and referred to as the
    trophic incoherence parameter $q$ by [1].

    Parameters
    ----------
    G : DiGraph
        A directed networkx graph

    cannibalism: Boolean
        If set to False, self edges are not considered in the calculation

    Returns
    -------
    trophic_incoherence_parameter : float
        The trophic coherence of a graph

    References
    ----------
    .. [1] Samuel Johnson, Virginia Dominguez-Garcia, Luca Donetti, Miguel A.
        Munoz (2014) PNAS "Trophic coherence determines food-web stability"
    """
    import numpy as np
    if cannibalism:
        diffs = trophic_differences(G, weight=weight)
    else:
        self_loops = list(nx.selfloop_edges(G))
        if self_loops:
            G_2 = G.copy()
            G_2.remove_edges_from(self_loops)
        else:
            G_2 = G
        diffs = trophic_differences(G_2, weight=weight)
    return np.std(list(diffs.values()))
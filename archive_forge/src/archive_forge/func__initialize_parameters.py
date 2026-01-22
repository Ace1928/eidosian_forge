import collections
import networkx as nx
def _initialize_parameters(G1, G2, G2_degree, node_label=None, default_label=-1):
    """Initializes all the necessary parameters for VF2++

    Parameters
    ----------
    G1,G2: NetworkX Graph or MultiGraph instances.
        The two graphs to check for isomorphism or monomorphism

    G1_labels,G2_labels: dict
        The label of every node in G1 and G2 respectively

    Returns
    -------
    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2
        G1_labels,G2_labels: dict

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_out, T2_out: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti
    """
    G1_labels = dict(G1.nodes(data=node_label, default=default_label))
    G2_labels = dict(G2.nodes(data=node_label, default=default_label))
    graph_params = _GraphParameters(G1, G2, G1_labels, G2_labels, nx.utils.groups(G1_labels), nx.utils.groups(G2_labels), nx.utils.groups(G2_degree))
    T1, T1_in = (set(), set())
    T2, T2_in = (set(), set())
    if G1.is_directed():
        T1_tilde, T1_tilde_in = (set(G1.nodes()), set())
        T2_tilde, T2_tilde_in = (set(G2.nodes()), set())
    else:
        T1_tilde, T1_tilde_in = (set(G1.nodes()), set())
        T2_tilde, T2_tilde_in = (set(G2.nodes()), set())
    state_params = _StateParameters({}, {}, T1, T1_in, T1_tilde, T1_tilde_in, T2, T2_in, T2_tilde, T2_tilde_in)
    return (graph_params, state_params)
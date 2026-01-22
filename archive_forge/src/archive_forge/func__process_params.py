import networkx as nx
from networkx.utils import np_random_state
def _process_params(G, center, dim):
    import numpy as np
    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph
    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)
    if len(center) != dim:
        msg = 'length of center coordinates must match dimension of layout'
        raise ValueError(msg)
    return (G, center)
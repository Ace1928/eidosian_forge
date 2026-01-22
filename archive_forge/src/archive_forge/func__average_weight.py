import networkx as nx
from networkx.utils import pairwise
def _average_weight(G, path, weight=None):
    """Returns the average weight of an edge in a weighted path.

    Parameters
    ----------
    G : graph
      A networkx graph.

    path: list
      A list of vertices that define the path.

    weight : None or string, optional (default=None)
      If None, edge weights are ignored.  Then the average weight of an edge
      is assumed to be the multiplicative inverse of the length of the path.
      Otherwise holds the name of the edge attribute used as weight.
    """
    path_length = len(path) - 1
    if path_length <= 0:
        return 0
    if weight is None:
        return 1 / path_length
    total_weight = sum((G.edges[i, j][weight] for i, j in pairwise(path)))
    return total_weight / path_length
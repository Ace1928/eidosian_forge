from collections import deque
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for
@not_implemented_for('directed')
def _bfs_with_marks(G, start_node, check_set):
    """Breadth-first-search with markings.

    Performs BFS starting from ``start_node`` and whenever a node
    inside ``check_set`` is met, it is "marked". Once a node is marked,
    BFS does not continue along that path. The resulting marked nodes
    are returned.

    Parameters
    ----------
    G : nx.Graph
        An undirected graph.
    start_node : node
        The start of the BFS.
    check_set : set
        The set of nodes to check against.

    Returns
    -------
    marked : set
        A set of nodes that were marked.
    """
    visited = {}
    marked = set()
    queue = []
    visited[start_node] = None
    queue.append(start_node)
    while queue:
        m = queue.pop(0)
        for nbr in G.neighbors(m):
            if nbr not in visited:
                visited[nbr] = None
                if nbr in check_set:
                    marked.add(nbr)
                else:
                    queue.append(nbr)
    return marked
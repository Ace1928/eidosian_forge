from collections import defaultdict
from functools import partial
from itertools import combinations
import networkx as nx
from networkx.utils import arbitrary_element
from networkx.utils.decorators import not_implemented_for
def _find_partition(G, starting_cell):
    """Find a partition of the vertices of G into cells of complete graphs

    Parameters
    ----------
    G : NetworkX Graph
    starting_cell : tuple of vertices in G which form a cell

    Returns
    -------
    List of tuples of vertices of G

    Raises
    ------
    NetworkXError
        If a cell is not a complete subgraph then G is not a line graph
    """
    G_partition = G.copy()
    P = [starting_cell]
    G_partition.remove_edges_from(list(combinations(starting_cell, 2)))
    partitioned_vertices = list(starting_cell)
    while G_partition.number_of_edges() > 0:
        u = partitioned_vertices.pop()
        deg_u = len(G_partition[u])
        if deg_u != 0:
            new_cell = [u] + list(G_partition[u])
            for u in new_cell:
                for v in new_cell:
                    if u != v and v not in G_partition[u]:
                        msg = 'G is not a line graph (partition cell not a complete subgraph)'
                        raise nx.NetworkXError(msg)
            P.append(tuple(new_cell))
            G_partition.remove_edges_from(list(combinations(new_cell, 2)))
            partitioned_vertices += new_cell
    return P
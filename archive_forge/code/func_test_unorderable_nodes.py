import pytest
import networkx as nx
from networkx.utils import pairwise
def test_unorderable_nodes(self):
    """Tests that A* accommodates nodes that are not orderable.

        For more information, see issue #554.

        """
    nodes = [object() for n in range(4)]
    G = nx.Graph()
    G.add_edges_from(pairwise(nodes, cyclic=True))
    path = nx.astar_path(G, nodes[0], nodes[2])
    assert len(path) == 3
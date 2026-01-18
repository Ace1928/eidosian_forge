import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_trivial_path(self):
    """Tests that the trivial path, a path of length one, is
        considered a simple path in a graph.

        """
    G = nx.trivial_graph()
    assert nx.is_simple_path(G, [0])
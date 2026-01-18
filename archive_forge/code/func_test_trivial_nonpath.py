import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_trivial_nonpath(self):
    """Tests that a list whose sole element is an object not in the
        graph is not considered a simple path.

        """
    G = nx.trivial_graph()
    assert not nx.is_simple_path(G, ['not a node'])
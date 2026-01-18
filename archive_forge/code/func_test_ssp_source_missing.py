import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_ssp_source_missing():
    with pytest.raises(nx.NodeNotFound):
        G = nx.Graph()
        nx.add_path(G, [1, 2, 3])
        list(nx.shortest_simple_paths(G, 0, 3))
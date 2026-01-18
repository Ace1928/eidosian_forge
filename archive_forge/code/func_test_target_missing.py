import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_target_missing():
    with pytest.raises(nx.NodeNotFound):
        G = nx.Graph()
        nx.add_path(G, [1, 2, 3])
        list(nx.all_simple_paths(nx.MultiGraph(G), 1, 4))
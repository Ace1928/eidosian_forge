import pytest
import networkx as nx
from networkx.utils import pairwise
def test_path_length_no_sources(self):
    with pytest.raises(ValueError):
        nx.multi_source_dijkstra_path_length(nx.Graph(), {})
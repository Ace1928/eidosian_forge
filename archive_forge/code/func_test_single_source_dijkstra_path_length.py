import pytest
import networkx as nx
from networkx.utils import pairwise
def test_single_source_dijkstra_path_length(self):
    pl = nx.single_source_dijkstra_path_length
    assert dict(pl(self.MXG4, 0))[2] == 4
    spl = pl(self.MXG4, 0, cutoff=2)
    assert 2 not in spl
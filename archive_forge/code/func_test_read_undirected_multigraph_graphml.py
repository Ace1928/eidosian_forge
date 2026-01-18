import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_read_undirected_multigraph_graphml(self):
    G = self.undirected_multigraph
    H = nx.read_graphml(self.undirected_multigraph_fh)
    assert nodes_equal(G.nodes(), H.nodes())
    assert edges_equal(G.edges(), H.edges())
    self.undirected_multigraph_fh.seek(0)
    PG = nx.parse_graphml(self.undirected_multigraph_data)
    assert nodes_equal(G.nodes(), PG.nodes())
    assert edges_equal(G.edges(), PG.edges())
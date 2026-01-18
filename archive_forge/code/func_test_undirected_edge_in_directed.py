import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_undirected_edge_in_directed(self):
    s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G" edgedefault=\'directed\'>\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <edge source="n0" target="n1"/>\n    <edge source="n1" target="n2" directed=\'false\'/>\n  </graph>\n</graphml>'
    fh = io.BytesIO(s.encode('UTF-8'))
    pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
    pytest.raises(nx.NetworkXError, nx.parse_graphml, s)
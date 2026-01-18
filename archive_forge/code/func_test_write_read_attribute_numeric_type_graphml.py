import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_write_read_attribute_numeric_type_graphml(self):
    from xml.etree.ElementTree import parse
    G = self.attribute_numeric_type_graph
    fh = io.BytesIO()
    self.writer(G, fh, infer_numeric_types=True)
    fh.seek(0)
    H = nx.read_graphml(fh)
    fh.seek(0)
    assert nodes_equal(G.nodes(), H.nodes())
    assert edges_equal(G.edges(), H.edges())
    assert edges_equal(G.edges(data=True), H.edges(data=True))
    self.attribute_numeric_type_fh.seek(0)
    xml = parse(fh)
    children = list(xml.getroot())
    assert len(children) == 3
    keys = [child.items() for child in children[:2]]
    assert len(keys) == 2
    assert ('attr.type', 'double') in keys[0]
    assert ('attr.type', 'double') in keys[1]
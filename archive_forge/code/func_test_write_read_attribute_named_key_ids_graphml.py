import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_write_read_attribute_named_key_ids_graphml(self):
    from xml.etree.ElementTree import parse
    G = self.attribute_named_key_ids_graph
    fh = io.BytesIO()
    self.writer(G, fh, named_key_ids=True)
    fh.seek(0)
    H = nx.read_graphml(fh)
    fh.seek(0)
    assert nodes_equal(G.nodes(), H.nodes())
    assert edges_equal(G.edges(), H.edges())
    assert edges_equal(G.edges(data=True), H.edges(data=True))
    self.attribute_named_key_ids_fh.seek(0)
    xml = parse(fh)
    children = list(xml.getroot())
    assert len(children) == 4
    keys = [child.items() for child in children[:3]]
    assert len(keys) == 3
    assert ('id', 'edge_prop') in keys[0]
    assert ('attr.name', 'edge_prop') in keys[0]
    assert ('id', 'prop2') in keys[1]
    assert ('attr.name', 'prop2') in keys[1]
    assert ('id', 'prop1') in keys[2]
    assert ('attr.name', 'prop1') in keys[2]
    default_behavior_fh = io.BytesIO()
    nx.write_graphml(G, default_behavior_fh)
    default_behavior_fh.seek(0)
    H = nx.read_graphml(default_behavior_fh)
    named_key_ids_behavior_fh = io.BytesIO()
    nx.write_graphml(G, named_key_ids_behavior_fh, named_key_ids=True)
    named_key_ids_behavior_fh.seek(0)
    J = nx.read_graphml(named_key_ids_behavior_fh)
    assert all((n1 == n2 for n1, n2 in zip(H.nodes, J.nodes)))
    assert all((e1 == e2 for e1, e2 in zip(H.edges, J.edges)))
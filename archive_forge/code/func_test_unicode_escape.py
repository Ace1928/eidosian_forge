import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_unicode_escape(self):
    import json
    a = {'a': '{"a": "123"}'}
    sa = json.dumps(a)
    G = nx.Graph()
    G.graph['test'] = sa
    fh = io.BytesIO()
    self.writer(G, fh)
    fh.seek(0)
    H = nx.read_graphml(fh)
    assert G.graph['test'] == H.graph['test']
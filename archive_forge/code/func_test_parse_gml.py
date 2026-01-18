import codecs
import io
import math
import os
import tempfile
from ast import literal_eval
from contextlib import contextmanager
from textwrap import dedent
import pytest
import networkx as nx
from networkx.readwrite.gml import literal_destringizer, literal_stringizer
def test_parse_gml(self):
    G = nx.parse_gml(self.simple_data, label='label')
    assert sorted(G.nodes()) == ['Node 1', 'Node 2', 'Node 3']
    assert sorted(G.edges()) == [('Node 1', 'Node 2'), ('Node 2', 'Node 3'), ('Node 3', 'Node 1')]
    assert sorted(G.edges(data=True)) == [('Node 1', 'Node 2', {'color': {'line': 'blue', 'thickness': 3}, 'label': 'Edge from node 1 to node 2'}), ('Node 2', 'Node 3', {'label': 'Edge from node 2 to node 3'}), ('Node 3', 'Node 1', {'label': 'Edge from node 3 to node 1'})]
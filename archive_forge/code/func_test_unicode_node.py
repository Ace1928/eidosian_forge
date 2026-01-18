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
def test_unicode_node(self):
    node = 'node' + chr(169)
    G = nx.Graph()
    G.add_node(node)
    fobj = tempfile.NamedTemporaryFile()
    nx.write_gml(G, fobj)
    fobj.seek(0)
    data = fobj.read().strip().decode('ascii')
    answer = 'graph [\n  node [\n    id 0\n    label "node&#169;"\n  ]\n]'
    assert data == answer
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
@pytest.mark.parametrize('stringizer', (None, literal_stringizer))
def test_tuplelabels(self, stringizer):
    G = nx.Graph()
    G.add_edge((0, 1), (1, 0))
    data = '\n'.join(nx.generate_gml(G, stringizer=stringizer))
    answer = 'graph [\n  node [\n    id 0\n    label "(0,1)"\n  ]\n  node [\n    id 1\n    label "(1,0)"\n  ]\n  edge [\n    source 0\n    target 1\n  ]\n]'
    assert data == answer
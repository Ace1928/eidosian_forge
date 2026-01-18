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
def test_reading_graph_with_list_property(self):
    with byte_file() as f:
        f.write(dedent('\n              graph [\n                node [\n                  id 0\n                  label "n1"\n                  properties "element"\n                  properties 0\n                  properties 1\n                  properties 2.5\n                ]\n              ]\n            ').encode('ascii'))
        f.seek(0)
        graph = nx.read_gml(f)
    assert graph.nodes(data=True)['n1'] == {'properties': ['element', 0, 1, 2.5]}
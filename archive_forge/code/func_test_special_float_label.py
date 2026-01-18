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
def test_special_float_label(self):
    special_floats = [float('nan'), float('+inf'), float('-inf')]
    try:
        import numpy as np
        special_floats += [np.nan, np.inf, np.inf * -1]
    except ImportError:
        special_floats += special_floats
    G = nx.cycle_graph(len(special_floats))
    attrs = dict(enumerate(special_floats))
    nx.set_node_attributes(G, attrs, 'nodefloat')
    edges = list(G.edges)
    attrs = {edges[i]: value for i, value in enumerate(special_floats)}
    nx.set_edge_attributes(G, attrs, 'edgefloat')
    fobj = tempfile.NamedTemporaryFile()
    nx.write_gml(G, fobj)
    fobj.seek(0)
    data = fobj.read().strip().decode('ascii')
    answer = 'graph [\n  node [\n    id 0\n    label "0"\n    nodefloat NAN\n  ]\n  node [\n    id 1\n    label "1"\n    nodefloat +INF\n  ]\n  node [\n    id 2\n    label "2"\n    nodefloat -INF\n  ]\n  node [\n    id 3\n    label "3"\n    nodefloat NAN\n  ]\n  node [\n    id 4\n    label "4"\n    nodefloat +INF\n  ]\n  node [\n    id 5\n    label "5"\n    nodefloat -INF\n  ]\n  edge [\n    source 0\n    target 1\n    edgefloat NAN\n  ]\n  edge [\n    source 0\n    target 5\n    edgefloat +INF\n  ]\n  edge [\n    source 1\n    target 2\n    edgefloat -INF\n  ]\n  edge [\n    source 2\n    target 3\n    edgefloat NAN\n  ]\n  edge [\n    source 3\n    target 4\n    edgefloat +INF\n  ]\n  edge [\n    source 4\n    target 5\n    edgefloat -INF\n  ]\n]'
    assert data == answer
    fobj.seek(0)
    graph = nx.read_gml(fobj)
    for indx, value in enumerate(special_floats):
        node_value = graph.nodes[str(indx)]['nodefloat']
        if math.isnan(value):
            assert math.isnan(node_value)
        else:
            assert node_value == value
        edge = edges[indx]
        string_edge = (str(edge[0]), str(edge[1]))
        edge_value = graph.edges[string_edge]['edgefloat']
        if math.isnan(value):
            assert math.isnan(edge_value)
        else:
            assert edge_value == value
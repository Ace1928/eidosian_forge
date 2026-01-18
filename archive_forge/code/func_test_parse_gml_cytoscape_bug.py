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
def test_parse_gml_cytoscape_bug(self):
    cytoscape_example = '\nCreator "Cytoscape"\nVersion 1.0\ngraph   [\n    node    [\n        root_index  -3\n        id  -3\n        graphics    [\n            x   -96.0\n            y   -67.0\n            w   40.0\n            h   40.0\n            fill    "#ff9999"\n            type    "ellipse"\n            outline "#666666"\n            outline_width   1.5\n        ]\n        label   "node2"\n    ]\n    node    [\n        root_index  -2\n        id  -2\n        graphics    [\n            x   63.0\n            y   37.0\n            w   40.0\n            h   40.0\n            fill    "#ff9999"\n            type    "ellipse"\n            outline "#666666"\n            outline_width   1.5\n        ]\n        label   "node1"\n    ]\n    node    [\n        root_index  -1\n        id  -1\n        graphics    [\n            x   -31.0\n            y   -17.0\n            w   40.0\n            h   40.0\n            fill    "#ff9999"\n            type    "ellipse"\n            outline "#666666"\n            outline_width   1.5\n        ]\n        label   "node0"\n    ]\n    edge    [\n        root_index  -2\n        target  -2\n        source  -1\n        graphics    [\n            width   1.5\n            fill    "#0000ff"\n            type    "line"\n            Line    [\n            ]\n            source_arrow    0\n            target_arrow    3\n        ]\n        label   "DirectedEdge"\n    ]\n    edge    [\n        root_index  -1\n        target  -1\n        source  -3\n        graphics    [\n            width   1.5\n            fill    "#0000ff"\n            type    "line"\n            Line    [\n            ]\n            source_arrow    0\n            target_arrow    3\n        ]\n        label   "DirectedEdge"\n    ]\n]\n'
    nx.parse_gml(cytoscape_example)
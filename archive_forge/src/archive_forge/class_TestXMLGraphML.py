import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
class TestXMLGraphML(TestWriteGraphML):
    writer = staticmethod(nx.write_graphml_xml)

    @classmethod
    def setup_class(cls):
        TestWriteGraphML.setup_class()
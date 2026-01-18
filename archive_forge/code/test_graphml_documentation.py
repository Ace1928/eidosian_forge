import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
Writing keys as edge id attributes means keys become strings.
        The original keys are stored as data, so read them back in
        if `str(key) == edge_id`
        This allows the adjacency to remain the same.
        
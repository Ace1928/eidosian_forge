import os
import tempfile
from io import StringIO
import pytest
import networkx as nx
from networkx.utils import graphs_equal

        Validate :mod:`pydot`-based usage of the passed NetworkX graph with the
        passed basename of an external GraphViz command (e.g., `dot`, `neato`).
        
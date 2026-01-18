import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
Tests that attempting to contract a nonexistent edge raises an
    exception.

    
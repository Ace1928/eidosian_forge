import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
Test that setting dtype int actually gives an integer array.

        For more information, see GitHub pull request #1363.

        
import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
Tests that a symmetric matrix has edges added only once to an
        undirected multigraph when using
        :func:`networkx.from_scipy_sparse_array`.

        
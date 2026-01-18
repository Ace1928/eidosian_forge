from itertools import permutations
import pytest
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
def test_partition_argument_non_integer_nodes():
    G = nx.Graph([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D')])
    partition = ({'A', 'B'}, {'C', 'D'})
    C = kernighan_lin_bisection(G, partition)
    assert_partition_equal(C, partition)
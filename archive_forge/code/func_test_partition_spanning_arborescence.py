import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_partition_spanning_arborescence():
    """
    Test that we can generate minimum spanning arborescences which respect the
    given partition.
    """
    G = nx.from_numpy_array(G_array, create_using=nx.DiGraph)
    G[3][0]['partition'] = nx.EdgePartition.EXCLUDED
    G[2][3]['partition'] = nx.EdgePartition.INCLUDED
    G[7][3]['partition'] = nx.EdgePartition.EXCLUDED
    G[0][2]['partition'] = nx.EdgePartition.EXCLUDED
    G[6][2]['partition'] = nx.EdgePartition.INCLUDED
    actual_edges = [(0, 4, 12), (1, 0, 4), (1, 5, 13), (2, 3, 21), (4, 7, 12), (5, 6, 14), (5, 8, 12), (6, 2, 21)]
    B = branchings.minimum_spanning_arborescence(G, partition='partition')
    assert_equal_branchings(build_branching(actual_edges), B)
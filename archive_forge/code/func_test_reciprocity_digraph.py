import pytest
import networkx as nx
def test_reciprocity_digraph(self):
    DG = nx.DiGraph([(1, 2), (2, 1)])
    reciprocity = nx.reciprocity(DG)
    assert reciprocity == 1.0
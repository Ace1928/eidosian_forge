import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.utils import pairwise
def are_edge_disjoint_paths(G, paths):
    if not paths:
        return False
    for path in paths:
        assert is_path(G, path)
    paths_edges = [list(pairwise(p)) for p in paths]
    num_of_edges = sum((len(e) for e in paths_edges))
    num_unique_edges = len(set.union(*[set(es) for es in paths_edges]))
    if num_of_edges == num_unique_edges:
        return True
    return False
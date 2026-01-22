import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.utils import pairwise
def are_node_disjoint_paths(G, paths):
    if not paths:
        return False
    for path in paths:
        assert is_path(G, path)
    st = {paths[0][0], paths[0][-1]}
    num_of_nodes = len([n for path in paths for n in path if n not in st])
    num_unique_nodes = len({n for path in paths for n in path if n not in st})
    if num_of_nodes == num_unique_nodes:
        return True
    return False
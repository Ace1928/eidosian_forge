import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_path__partition_provided_as_dict_of_sets():
    G = nx.path_graph(6)
    partition = {0: {0, 1}, 2: {2, 3}, 4: {4, 5}}
    M = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(M, [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    for n in M:
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 1
import pytest
import networkx as nx
from networkx.generators import directed
def test_reversed_subgraph_hash():
    """
    A directed graph with no bi-directional edges should yield different subgraph hashes
    to the same graph taken with edge directions reversed if there are no hash collisions.
    Here we test a cycle graph which is the minimal counterexample
    """
    G = nx.cycle_graph(5, create_using=nx.DiGraph)
    nx.set_node_attributes(G, {n: str(n) for n in G.nodes()}, name='label')
    G_reversed = G.reverse()
    h = nx.weisfeiler_lehman_subgraph_hashes(G, node_attr='label')
    h_reversed = nx.weisfeiler_lehman_subgraph_hashes(G_reversed, node_attr='label')
    assert h != h_reversed
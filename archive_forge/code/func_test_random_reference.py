import pytest
import random
import networkx as nx
from networkx import lattice_reference, omega, random_reference, sigma
def test_random_reference():
    G = nx.connected_watts_strogatz_graph(50, 6, 0.1, seed=rng)
    Gr = random_reference(G, niter=1, seed=rng)
    C = nx.average_clustering(G)
    Cr = nx.average_clustering(Gr)
    assert C > Cr
    with pytest.raises(nx.NetworkXError):
        next(random_reference(nx.Graph()))
    with pytest.raises(nx.NetworkXNotImplemented):
        next(random_reference(nx.DiGraph()))
    H = nx.Graph(((0, 1), (2, 3)))
    Hl = random_reference(H, niter=1, seed=rng)
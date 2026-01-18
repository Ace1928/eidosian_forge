import pytest
import random
import networkx as nx
from networkx import lattice_reference, omega, random_reference, sigma
def test_sigma():
    Gs = nx.connected_watts_strogatz_graph(50, 6, 0.1, seed=rng)
    Gr = nx.connected_watts_strogatz_graph(50, 6, 1, seed=rng)
    sigmas = sigma(Gs, niter=1, nrand=2, seed=rng)
    sigmar = sigma(Gr, niter=1, nrand=2, seed=rng)
    assert sigmar < sigmas
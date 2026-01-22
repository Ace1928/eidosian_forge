import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
def _generate_no_biconnected(max_attempts=50):
    attempts = 0
    while True:
        G = nx.fast_gnp_random_graph(100, 0.0575, seed=42)
        if nx.is_connected(G) and (not nx.is_biconnected(G)):
            attempts = 0
            yield G
        elif attempts >= max_attempts:
            msg = f'Tried {attempts} times: no suitable Graph.'
            raise Exception(msg)
        else:
            attempts += 1
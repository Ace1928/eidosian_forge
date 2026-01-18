import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_random_triad_deprecated():
    G = nx.path_graph(3, create_using=nx.DiGraph)
    with pytest.deprecated_call():
        nx.random_triad(G)
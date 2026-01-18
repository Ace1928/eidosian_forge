import networkx as nx
from networkx.utils import py_random_state
@py_random_state(1)
def weighted_choice(mapping, seed=None):
    """Returns a single element from a weighted sample.

    The input is a dictionary of items with weights as values.
    """
    rnd = seed.random() * sum(mapping.values())
    for k, w in mapping.items():
        rnd -= w
        if rnd < 0:
            return k
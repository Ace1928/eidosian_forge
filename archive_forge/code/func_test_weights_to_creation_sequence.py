import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_weights_to_creation_sequence(self):
    deg = [3, 2, 2, 1]
    with pytest.raises(ValueError):
        nxt.weights_to_creation_sequence(deg, with_labels=True, compact=True)
    assert nxt.weights_to_creation_sequence(deg, with_labels=True) == [(3, 'd'), (1, 'd'), (2, 'd'), (0, 'd')]
    assert nxt.weights_to_creation_sequence(deg, compact=True) == [4]
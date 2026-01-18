import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_uncompact(self):
    assert nxt.uncompact([3, 1, 2]) == ['d', 'd', 'd', 'i', 'd', 'd']
    assert nxt.uncompact(['d', 'd', 'i', 'd']) == ['d', 'd', 'i', 'd']
    assert nxt.uncompact(nxt.uncompact([(1, 'd'), (2, 'd'), (3, 'i'), (0, 'd')])) == nxt.uncompact([(1, 'd'), (2, 'd'), (3, 'i'), (0, 'd')])
    assert pytest.raises(TypeError, nxt.uncompact, [3.0, 1.0, 2.0])
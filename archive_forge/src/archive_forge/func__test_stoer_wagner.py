from itertools import chain
import pytest
import networkx as nx
def _test_stoer_wagner(G, answer, weight='weight'):
    cut_value, partition = nx.stoer_wagner(G, weight, heap=nx.utils.PairingHeap)
    assert cut_value == answer
    _check_partition(G, cut_value, partition, weight)
    cut_value, partition = nx.stoer_wagner(G, weight, heap=nx.utils.BinaryHeap)
    assert cut_value == answer
    _check_partition(G, cut_value, partition, weight)
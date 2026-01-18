import itertools
import random
import pytest
import networkx
import cirq
def test_wrapper_repr():
    q0 = cirq.LineQubit(0)
    node = cirq.contrib.CircuitDag.make_node(cirq.X(q0))
    expected = f'cirq.contrib.Unique({id(node)}, cirq.X(cirq.LineQubit(0)))'
    assert repr(node) == expected
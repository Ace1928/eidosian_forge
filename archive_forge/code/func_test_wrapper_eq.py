import itertools
import random
import pytest
import networkx
import cirq
def test_wrapper_eq():
    q0, q1 = cirq.LineQubit.range(2)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.contrib.CircuitDag.make_node(cirq.X(q0)))
    eq.add_equality_group(cirq.contrib.CircuitDag.make_node(cirq.X(q0)))
    eq.add_equality_group(cirq.contrib.CircuitDag.make_node(cirq.Y(q0)))
    eq.add_equality_group(cirq.contrib.CircuitDag.make_node(cirq.X(q1)))
import itertools
import random
import pytest
import networkx
import cirq
def test_two_identical_ops():
    q0 = cirq.LineQubit(0)
    dag = cirq.contrib.CircuitDag()
    dag.append(cirq.X(q0))
    dag.append(cirq.Y(q0))
    dag.append(cirq.X(q0))
    assert networkx.dag.is_directed_acyclic_graph(dag)
    assert len(dag.nodes()) == 3
    assert set(((n1.val, n2.val) for n1, n2 in dag.edges())) == {(cirq.X(q0), cirq.Y(q0)), (cirq.X(q0), cirq.X(q0)), (cirq.Y(q0), cirq.X(q0))}
from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_tagged_operation():
    q1 = cirq.GridQubit(1, 1)
    q2 = cirq.GridQubit(2, 2)
    op = cirq.X(q1).with_tags('tag1')
    op_repr = 'cirq.X(cirq.GridQubit(1, 1))'
    assert repr(op) == f"cirq.TaggedOperation({op_repr}, 'tag1')"
    assert op.qubits == (q1,)
    assert op.tags == ('tag1',)
    assert op.gate == cirq.X
    assert op.with_qubits(q2) == cirq.X(q2).with_tags('tag1')
    assert op.with_qubits(q2).qubits == (q2,)
    assert not cirq.is_measurement(op)
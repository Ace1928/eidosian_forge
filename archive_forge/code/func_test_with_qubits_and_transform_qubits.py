import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_with_qubits_and_transform_qubits():
    g = cirq.testing.ThreeQubitGate()
    g = cirq.testing.ThreeQubitGate()
    op = cirq.GateOperation(g, cirq.LineQubit.range(3))
    assert op.with_qubits(*cirq.LineQubit.range(3, 0, -1)) == cirq.GateOperation(g, cirq.LineQubit.range(3, 0, -1))
    assert op.transform_qubits(lambda e: cirq.LineQubit(-e.x)) == cirq.GateOperation(g, [cirq.LineQubit(0), cirq.LineQubit(-1), cirq.LineQubit(-2)])
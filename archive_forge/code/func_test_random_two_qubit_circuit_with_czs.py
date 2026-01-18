from typing import Optional, Dict, Sequence, Union, cast
import random
import numpy as np
import pytest
import cirq
import cirq.testing
def test_random_two_qubit_circuit_with_czs():
    num_czs = lambda circuit: len([o for o in circuit.all_operations() if isinstance(o.gate, cirq.CZPowGate)])
    c = cirq.testing.random_two_qubit_circuit_with_czs()
    assert num_czs(c) == 3
    assert {cirq.NamedQubit('q0'), cirq.NamedQubit('q1')} == c.all_qubits()
    assert all((isinstance(op.gate, cirq.PhasedXPowGate) for op in c[0].operations))
    assert c[0].qubits == c.all_qubits()
    c = cirq.testing.random_two_qubit_circuit_with_czs(num_czs=0)
    assert num_czs(c) == 0
    assert {cirq.NamedQubit('q0'), cirq.NamedQubit('q1')} == c.all_qubits()
    assert all((isinstance(op.gate, cirq.PhasedXPowGate) for op in c[0].operations))
    assert c[0].qubits == c.all_qubits()
    a, b = cirq.LineQubit.range(2)
    c = cirq.testing.random_two_qubit_circuit_with_czs(num_czs=1, q1=b)
    assert num_czs(c) == 1
    assert {b, cirq.NamedQubit('q0')} == c.all_qubits()
    assert all((isinstance(op.gate, cirq.PhasedXPowGate) for op in c[0].operations))
    assert c[0].qubits == c.all_qubits()
    c = cirq.testing.random_two_qubit_circuit_with_czs(num_czs=2, q0=a)
    assert num_czs(c) == 2
    assert {a, cirq.NamedQubit('q1')} == c.all_qubits()
    assert all((isinstance(op.gate, cirq.PhasedXPowGate) for op in c[0].operations))
    assert c[0].qubits == c.all_qubits()
    c = cirq.testing.random_two_qubit_circuit_with_czs(num_czs=3, q0=a, q1=b)
    assert num_czs(c) == 3
    assert c.all_qubits() == {a, b}
    assert all((isinstance(op.gate, cirq.PhasedXPowGate) for op in c[0].operations))
    assert c[0].qubits == c.all_qubits()
    seed = 77
    c1 = cirq.testing.random_two_qubit_circuit_with_czs(num_czs=4, q0=a, q1=b, random_state=seed)
    assert num_czs(c1) == 4
    assert c1.all_qubits() == {a, b}
    c2 = cirq.testing.random_two_qubit_circuit_with_czs(num_czs=4, q0=a, q1=b, random_state=seed)
    assert c1 == c2
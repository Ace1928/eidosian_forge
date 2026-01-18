import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_repr_preserves_qubit_order():
    q0, q1, q2 = _make_qubits(3)
    pauli_string = cirq.PauliString({q2: cirq.X, q1: cirq.Y, q0: cirq.Z})
    assert eval(repr(pauli_string)).qubits == pauli_string.qubits
    pauli_string = cirq.PauliString(cirq.X(q2), cirq.Y(q1), cirq.Z(q0))
    assert eval(repr(pauli_string)).qubits == pauli_string.qubits
    pauli_string = cirq.PauliString(cirq.Z(q0), cirq.Y(q1), cirq.X(q2))
    assert eval(repr(pauli_string)).qubits == pauli_string.qubits
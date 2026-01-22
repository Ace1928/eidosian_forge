import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def _small_sample_qubit_pauli_maps():
    """A few representative samples of qubit maps.

    Only tests 10 combinations of Paulis to speed up testing.
    """
    qubits = _make_qubits(3)
    yield {}
    yield {qubits[0]: cirq.X}
    yield {qubits[1]: cirq.X}
    yield {qubits[2]: cirq.X}
    yield {qubits[1]: cirq.Z}
    yield {qubits[0]: cirq.Y, qubits[1]: cirq.Z}
    yield {qubits[1]: cirq.Z, qubits[2]: cirq.X}
    yield {qubits[0]: cirq.X, qubits[1]: cirq.X, qubits[2]: cirq.X}
    yield {qubits[0]: cirq.X, qubits[1]: cirq.Y, qubits[2]: cirq.Z}
    yield {qubits[0]: cirq.Z, qubits[1]: cirq.X, qubits[2]: cirq.Y}
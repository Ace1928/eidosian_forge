import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_with_coefficient():
    qubits = cirq.LineQubit.range(4)
    qubit_pauli_map = {q: cirq.Pauli.by_index(q.x) for q in qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, 1.23)
    ps2 = pauli_string.with_coefficient(1.0)
    assert ps2.coefficient == 1.0
    assert ps2.equal_up_to_coefficient(pauli_string)
    assert pauli_string != ps2
    assert pauli_string.coefficient == 1.23
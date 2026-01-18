import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_pass_unsupported_operations_over():
    q0, = _make_qubits(1)
    pauli_string = cirq.PauliString({q0: cirq.X})
    with pytest.raises(TypeError, match='not a known Clifford'):
        pauli_string.pass_operations_over([cirq.T(q0)])
import itertools
from typing import Sequence
import numpy as np
import pytest
import cirq
def test_xeb_fidelity_invalid_bitstrings():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    bitstrings = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        cirq.xeb_fidelity(circuit, bitstrings, (q0, q1))
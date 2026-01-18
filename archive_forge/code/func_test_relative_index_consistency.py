import itertools
import numpy as np
import pytest
import cirq
def test_relative_index_consistency():
    for pauli_1 in (cirq.X, cirq.Y, cirq.Z):
        for pauli_2 in (cirq.X, cirq.Y, cirq.Z):
            shift = pauli_2.relative_index(pauli_1)
            assert cirq.Pauli.by_relative_index(pauli_1, shift) == pauli_2
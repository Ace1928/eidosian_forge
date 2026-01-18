import itertools
import numpy as np
import pytest
import cirq
def test_phased_pauli_product():
    assert cirq.X.phased_pauli_product(cirq.I) == (1, cirq.X)
    assert cirq.X.phased_pauli_product(cirq.X) == (1, cirq.I)
    assert cirq.X.phased_pauli_product(cirq.Y) == (1j, cirq.Z)
    assert cirq.X.phased_pauli_product(cirq.Z) == (-1j, cirq.Y)
    assert cirq.Y.phased_pauli_product(cirq.I) == (1, cirq.Y)
    assert cirq.Y.phased_pauli_product(cirq.X) == (-1j, cirq.Z)
    assert cirq.Y.phased_pauli_product(cirq.Y) == (1, cirq.I)
    assert cirq.Y.phased_pauli_product(cirq.Z) == (1j, cirq.X)
    assert cirq.Z.phased_pauli_product(cirq.I) == (1, cirq.Z)
    assert cirq.Z.phased_pauli_product(cirq.X) == (1j, cirq.Y)
    assert cirq.Z.phased_pauli_product(cirq.Y) == (-1j, cirq.X)
    assert cirq.Z.phased_pauli_product(cirq.Z) == (1, cirq.I)
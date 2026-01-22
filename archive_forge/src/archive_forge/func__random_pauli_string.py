import functools
import operator
import numpy as np
import pytest
import cirq
import cirq.contrib.quimb as ccq
def _random_pauli_string(qubits, rs, coefficients=False):
    ps = cirq.PauliString(({q: p} for q, p in zip(qubits, rs.choice([cirq.X, cirq.Y, cirq.Z, cirq.I], size=len(qubits)))))
    if coefficients:
        return rs.uniform(-1, 1) * ps
    return ps
import itertools
import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('gate', _all_interaction_gates())
def test_pauli_interaction_gates_consistent_protocols(gate):
    cirq.testing.assert_implements_consistent_protocols(gate)
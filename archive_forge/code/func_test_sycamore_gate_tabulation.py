import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.heuristic_decompositions.two_qubit_gate_tabulation import (
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
from cirq.testing import random_special_unitary, assert_equivalent_repr
@pytest.mark.parametrize('seed', [0, 1])
def test_sycamore_gate_tabulation(seed):
    base_gate = cirq.unitary(cirq.FSimGate(np.pi / 2, np.pi / 6))
    tab = two_qubit_gate_product_tabulation(base_gate, 0.1, sample_scaling=2, random_state=np.random.RandomState(seed))
    result = tab.compile_two_qubit_gate(base_gate)
    assert result.success
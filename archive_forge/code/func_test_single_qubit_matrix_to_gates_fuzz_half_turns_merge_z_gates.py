import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('pre_turns,post_turns', [(random.random(), random.random()) for _ in range(10)])
def test_single_qubit_matrix_to_gates_fuzz_half_turns_merge_z_gates(pre_turns, post_turns):
    intended_effect = cirq.dot(cirq.unitary(cirq.Z ** (2 * pre_turns)), cirq.unitary(cirq.X), cirq.unitary(cirq.Z ** (2 * post_turns)))
    gates = cirq.single_qubit_matrix_to_gates(intended_effect, tolerance=1e-07)
    assert len(gates) <= 2
    assert_gates_implement_unitary(gates, intended_effect, atol=1e-06)